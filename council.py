"""AI Council - idea generation, debate and evaluation.

A modern, provider-agnostic rewrite of the original ``openaicouncil.py``.

Key differences from the original:
  * Uses Chat Completions (works with OpenRouter *and* OpenAI) instead of the
    now-deprecated OpenAI Assistants API.
  * A multi-judge panel with distinct temperaments returns structured 1-10
    rubric scores (median-aggregated) instead of the crude VADER sentiment proxy.
  * Ideas are ranked by a direct post-debate verdict on the idea's own merit,
    not by how much total argument they generated.
  * Runs are reproducible: a --seed is recorded and forwarded to the API.
  * Each persona keeps its own running conversation per side, so members build
    on their prior arguments.
  * Fully configurable from the command line; results are written as JSON.

Usage
-----
    python council.py --topic "How to boost local civic participation"
    python council.py --topic "..." --provider openai --model gpt-4o
    python council.py --topic "..." --tags economics innovation --ideas 6

Run ``python council.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

from llm import LLMClient, LLMConfig


MEMBERS_DIR = "CouncilMembers"


def pmap(fn, items, workers: int):
    """Map ``fn`` over ``items`` concurrently with threads, preserving order.

    Threads (not asyncio) because the OpenAI SDK is synchronous and releases the
    GIL during network I/O, so debate/scoring calls that are independent run in
    parallel. Falls back to a serial map for tiny workloads.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


class Side(Enum):
    FOR = "for"
    AGAINST = "against"


# --------------------------------------------------------------------------- #
# Persona loading
# --------------------------------------------------------------------------- #
@dataclass
class Persona:
    name: str
    instructions: str
    tags: list[str]


def load_personas(tags: list[str], require_all: bool) -> list[Persona]:
    """Load council member personas from ``CouncilMembers/``.

    ``tags`` filters members by the comma-separated tags in their ``info.txt``.
    With ``require_all`` every requested tag must be present (AND); otherwise
    any single match is enough (OR). An empty tag list loads everyone.
    """
    personas: list[Persona] = []
    if not os.path.isdir(MEMBERS_DIR):
        raise FileNotFoundError(f"'{MEMBERS_DIR}' directory not found.")

    for entry in sorted(os.listdir(MEMBERS_DIR)):
        folder = os.path.join(MEMBERS_DIR, entry)
        if not os.path.isdir(folder) or entry.lower() == "judge":
            continue

        instr_path = os.path.join(folder, "instructions.txt")
        info_path = os.path.join(folder, "info.txt")
        if not os.path.exists(instr_path):
            continue

        member_tags: list[str] = []
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as f:
                member_tags = [t.strip() for t in f.read().split(",") if t.strip()]

        if tags:
            present = [t for t in tags if t in member_tags]
            if require_all and len(present) != len(tags):
                continue
            if not require_all and not present:
                continue

        with open(instr_path, encoding="utf-8") as f:
            instructions = f.read().strip()

        personas.append(Persona(name=entry, instructions=instructions, tags=member_tags))

    return personas


def load_judge_instructions() -> str:
    path = os.path.join(MEMBERS_DIR, "Judge", "instructions.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return (
        "You are an impartial, rigorous judge. Weigh the logic, evidence and "
        "rationality of arguments and score them fairly."
    )


# --------------------------------------------------------------------------- #
# Persona knowledge cards
# --------------------------------------------------------------------------- #
# A compact, always-on "stay recognisably yourself" block (signature concepts,
# vocabulary, characteristic stances, intellectual rivals) injected into a
# member's system prompt. Distinct from RAG, which retrieves query-relevant
# passages per turn; the card is persistent flavour. Stored as card.json in the
# member folder (not under Files/, so it is not part of the RAG corpus). Generate
# them with cards.py.
_CARD_FIELDS = [
    ("signature_concepts", "Concepts you are known for"),
    ("key_terms", "Vocabulary you naturally use"),
    ("stances", "Positions you characteristically hold"),
    ("rivals", "Ideas or thinkers you push against"),
]


def load_card(name: str) -> dict | None:
    path = os.path.join(MEMBERS_DIR, name, "card.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def format_card(card: dict | None) -> str:
    if not card:
        return ""
    lines = []
    for key, label in _CARD_FIELDS:
        vals = [str(v).strip() for v in (card.get(key) or []) if str(v).strip()]
        if vals:
            lines.append(f"- {label}: " + "; ".join(vals))
    if not lines:
        return ""
    return (
        "\n\nKnowledge card — stay recognisably yourself, drawing naturally on "
        "these (don't just list them):\n" + "\n".join(lines)
    )


# --------------------------------------------------------------------------- #
# Debating members
# --------------------------------------------------------------------------- #
# Many council members thought and wrote in languages other than English
# (Arabic, Greek, Chinese, French, German, Russian, Italian, ...). Rather than
# debating in their native tongue -- which degrades reasoning for classical/low
# resource languages and makes scoring unfair and transcripts unreadable -- they
# reason in English but code-switch: native key terms are kept as precise
# conceptual anchors, always glossed in English. This is a no-op for figures who
# already thought in English.
CODE_SWITCH_DIRECTIVE = (
    "\n\nLanguage: Reason and argue in English so the panel and judges can "
    "follow you. However, if you originally thought and wrote in another "
    "language, weave your authentic original-language key terms into your "
    "arguments wherever they carry a nuance that English flattens, and "
    "immediately gloss each in English on first use (for example: "
    "\"'asabiyyah' - group solidarity\"). Use these terms as sharp conceptual "
    "anchors; do NOT write whole passages in another language."
)


class Member:
    """A persona that argues one side and remembers its own contributions.

    If given a ``retriever`` (a ``rag.CorpusIndex``), each prompt is augmented
    with the passages from the member's own writings most relevant to the query,
    so arguments are grounded in that member's corpus.
    """

    def __init__(self, persona: Persona, side: Side, client: LLMClient,
                 retriever=None, rag_k: int = 4, card_text: str = ""):
        self.persona = persona
        self.side = side
        self.client = client
        self.retriever = retriever
        self.rag_k = rag_k
        system_prompt = persona.instructions + CODE_SWITCH_DIRECTIVE + card_text
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def _augment(self, prompt: str, query: str) -> str:
        if not self.retriever:
            return prompt
        chunks = self.retriever.retrieve(query, self.rag_k)
        if not chunks:
            return prompt
        refs = "\n---\n".join(f"[{c.source}] {c.text}" for c in chunks)
        return (
            "Relevant passages retrieved from your reference corpus (your own "
            "writings and notes on your ideas). Ground your response in these "
            "where apt, drawing on their specific ideas and language (do not "
            "quote at length):\n\"\"\"\n"
            f"{refs}\n\"\"\"\n\n{prompt}"
        )

    def _ask(self, prompt: str, query: str | None = None) -> str:
        content = self._augment(prompt, query) if query else prompt
        self.history.append({"role": "user", "content": content})
        reply = self.client.chat(self.history)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def generate_idea(self, topic: str) -> str:
        prompt = (
            f"Topic: {topic}\n\n"
            "Drawing on your distinctive expertise, perspective and body of "
            "work, propose ONE original, specific and interesting idea to "
            "address this topic. Give it a short title, then 3-5 sentences of "
            "explanation grounded in your worldview."
        )
        return self._ask(prompt, query=topic)

    def argue(self, topic: str, idea: str) -> str:
        """Opening statement for this member's side."""
        stance = "argue in favour of" if self.side is Side.FOR else "argue against"
        prompt = (
            f"Topic: {topic}\nIdea under debate: {idea}\n\n"
            f"This is your OPENING STATEMENT. Using your expertise, {stance} this "
            "idea. Be concise, concrete and persuasive. Lead with your strongest "
            "point."
        )
        return self._ask(prompt, query=f"{topic} {idea}")

    def rebut(self, topic: str, idea: str, opponent_arg: str) -> str:
        """Cross-examination turn: rebut the opponent's latest statement."""
        prompt = (
            f"Topic: {topic}\nIdea under debate: {idea}\n\n"
            f"Your opponent just argued:\n\"{opponent_arg}\"\n\n"
            "Cross-examine them: write a sharp, concise rebuttal from your side "
            "that exposes the weaknesses in their reasoning while reinforcing your "
            "position. Engage their specific points; do not merely repeat yourself."
        )
        return self._ask(prompt, query=f"{topic} {idea} {opponent_arg}")

    def closing(self, topic: str, idea: str) -> str:
        """Closing statement, drawing on the member's own debate history."""
        favour = "should be adopted" if self.side is Side.FOR else "should be rejected"
        prompt = (
            f"Topic: {topic}\nIdea under debate: {idea}\n\n"
            "This is your CLOSING STATEMENT. Weighing the debate so far, deliver "
            f"the single most compelling reason the idea {favour}. Be brief and "
            "decisive; consolidate your case rather than introducing weak new points."
        )
        return self._ask(prompt, query=f"{topic} {idea}")


# --------------------------------------------------------------------------- #
# Judges
# --------------------------------------------------------------------------- #
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)

# A panel of judges with distinct temperaments reduces the variance of any one
# judge. For a panel of N, the first N temperaments are used (cycling if N is
# larger than the list).
JUDGE_TEMPERAMENTS: list[tuple[str, str]] = [
    (
        "Sceptic",
        "You are the SCEPTIC on the judging panel: ruthlessly critical. Demand "
        "airtight logic and hard evidence, and dock heavily for any fallacy, "
        "hand-waving or unsupported leap.",
    ),
    (
        "Pragmatist",
        "You are the PRAGMATIST on the judging panel: you reward reasoning that "
        "is constructive, actionable and grounded in real-world feasibility, "
        "while still penalising sloppy logic.",
    ),
    (
        "Empiricist",
        "You are the EMPIRICIST on the judging panel: you weight concrete "
        "evidence, data and historical precedent above rhetoric, and reward "
        "claims that could be tested or have been borne out in practice.",
    ),
    (
        "Theorist",
        "You are the THEORIST on the judging panel: you prize internal "
        "coherence, conceptual clarity and first-principles rigour, rewarding "
        "arguments that are logically well-structured and consistent.",
    ),
    (
        "Generalist",
        "You are the GENERALIST on the judging panel: a balanced, impartial "
        "adjudicator who weighs logic, evidence and clarity together without "
        "favouring any one dimension.",
    ),
]

_NATIVE_NOTE = (
    "\n\nArguments and ideas may include original-language key terms (glossed "
    "in English) from thinkers who worked in other languages; judge the "
    "substance and do not reward or penalise them for it."
    "\n\nAlways reply with a single JSON object and nothing else."
)


# --------------------------------------------------------------------------- #
# Multi-dimensional rubric for judging an idea's own merit. Each axis is scored
# 1-10 (higher is better) and combined into a weighted overall verdict; users
# can re-weight the axes (e.g. --weights novelty=2,feasibility=1.5).
# --------------------------------------------------------------------------- #
@dataclass
class Axis:
    name: str
    description: str
    weight: float = 1.0


DEFAULT_IDEA_RUBRIC: list[Axis] = [
    Axis("novelty", "originality and non-obviousness of the idea"),
    Axis("feasibility", "how practically implementable it is with real resources"),
    Axis("evidence", "how well supported by evidence, data, precedent or theory"),
    Axis("logic", "internal coherence and soundness once scrutinised"),
    Axis("risk", "downside safety: 10 = robust with limited downside, 1 = fragile or dangerous"),
]


def build_rubric(weights: dict[str, float] | None) -> list[Axis]:
    """Copy the default rubric, overriding weights for named axes."""
    weights = weights or {}
    return [Axis(a.name, a.description, float(weights.get(a.name, a.weight)))
            for a in DEFAULT_IDEA_RUBRIC]


class Judge:
    """A single judge with a temperament. Scores arguments with one number and
    ideas across a multi-axis rubric."""

    def __init__(
        self,
        instructions: str,
        client: LLMClient,
        name: str = "Judge",
        temperament: str = "",
    ):
        self.client = client
        self.name = name
        self.instructions = instructions
        if temperament:
            self.instructions += "\n\n" + temperament
        self.instructions += _NATIVE_NOTE

    def _judge(self, user_prompt: str) -> dict:
        raw = self.client.chat(
            [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"_raw": raw}
        except (json.JSONDecodeError, TypeError):
            return {"_raw": raw}

    def score_argument(self, topic: str, idea: str, argument: str) -> tuple[float, str]:
        data = self._judge(
            f"Topic: {topic}\nIdea: {idea}\n\n"
            f"Argument to judge:\n\"{argument}\"\n\n"
            "Score its logical strength, evidence and rigour, where 10 is a "
            "flawless, rigorous, well-evidenced argument and 1 is fallacious or "
            "baseless. Reply with JSON "
            '{"score": <integer 1-10>, "reasoning": "<one or two sentences>"}.'
        )
        return _num(data.get("score"), data), str(data.get("reasoning", "")).strip()

    def score_idea(
        self, topic: str, idea: str, for_case: str, against_case: str, rubric: list[Axis]
    ) -> tuple[dict, str]:
        axes_desc = "\n".join(f"- {a.name}: {a.description}" for a in rubric)
        keys = ", ".join(f'"{a.name}": <integer 1-10>' for a in rubric)
        data = self._judge(
            f"Topic: {topic}\n\nIdea under judgement:\n\"{idea}\"\n\n"
            f"The strongest points made FOR it:\n{for_case}\n\n"
            f"The strongest points made AGAINST it:\n{against_case}\n\n"
            "Having weighed both sides, score the IDEA ITSELF (not the eloquence "
            "of either side) on each of these axes, 1-10 where higher is better:\n"
            f"{axes_desc}\n\n"
            f'Reply with JSON {{"scores": {{{keys}}}, "reasoning": "<one or two '
            'sentences>"}.'
        )
        raw_scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
        scores = {a.name: _num(raw_scores.get(a.name), data) for a in rubric}
        return scores, str(data.get("reasoning", "")).strip()


def _num(value, data: dict) -> float:
    """Coerce a judge score to a float, falling back to a regex over the raw
    text and finally to a neutral 5.0 so one malformed reply can't crash a run."""
    try:
        return float(value)
    except (TypeError, ValueError):
        match = _SCORE_RE.search(data.get("_raw", "") or json.dumps(data))
        return float(match.group(1)) if match else 5.0


class JudgePanel:
    """Aggregates several judges; the aggregate is the median (robust to a
    single outlier judge)."""

    def __init__(self, instructions: str, client: LLMClient, n_judges: int = 3):
        n_judges = max(1, n_judges)
        self.judges = [
            Judge(instructions, client, name=JUDGE_TEMPERAMENTS[i % len(JUDGE_TEMPERAMENTS)][0],
                  temperament=JUDGE_TEMPERAMENTS[i % len(JUDGE_TEMPERAMENTS)][1])
            for i in range(n_judges)
        ]

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.judges]

    def score_argument(self, topic: str, idea: str, argument: str) -> dict:
        votes = []
        for j in self.judges:
            score, reasoning = j.score_argument(topic, idea, argument)
            votes.append({"judge": j.name, "score": score, "reasoning": reasoning})
        return {"aggregate": statistics.median(v["score"] for v in votes), "votes": votes}

    def score_idea(
        self, topic: str, idea: str, for_case: str, against_case: str, rubric: list[Axis]
    ) -> dict:
        votes = []
        for j in self.judges:
            scores, reasoning = j.score_idea(topic, idea, for_case, against_case, rubric)
            votes.append({"judge": j.name, "scores": scores, "reasoning": reasoning})
        # Median per axis across the panel, then a weighted overall verdict.
        axes = {
            a.name: statistics.median(v["scores"][a.name] for v in votes)
            for a in rubric
        }
        total_w = sum(a.weight for a in rubric) or 1.0
        overall = sum(axes[a.name] * a.weight for a in rubric) / total_w
        return {"overall": round(overall, 2), "axes": axes, "votes": votes}


# --------------------------------------------------------------------------- #
# Council orchestration
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    topic: str
    tags: list[str] = field(default_factory=list)
    require_all: bool = False
    ideas_to_evaluate: int = 3
    provider: str = "openrouter"
    model: str | None = None
    temperature: float = 0.8
    judges: int = 3
    rounds: int = 1
    closing: bool = True
    seed: int | None = None
    neutralize: bool = True
    concurrency: int = 8
    dedupe: bool = True
    dedupe_threshold: float = 0.85
    cache: bool = True
    rag: bool = True
    rag_k: int = 4
    cards: bool = True
    weights: dict = field(default_factory=dict)
    output: str = "results.json"


# How many of the top-scoring arguments per side feed the idea verdict. Keeps
# the verdict prompt bounded regardless of council size.
_TOP_ARGS_PER_SIDE = 3


class Council:
    def __init__(self, run: RunConfig, progress_cb=None):
        self.run = run
        # Optional progress sink: called with {"type": "progress", "frac": 0..1,
        # "msg": str}. Used by the web UI to stream live progress. Thread-safe by
        # contract (the UI pushes events onto a queue), so it is safe to call
        # from the internal thread pool.
        self._progress_cb = progress_cb or (lambda ev: None)
        # Reproducibility: seed Python RNG and forward the seed to the API.
        if run.seed is not None:
            random.seed(run.seed)
        self.client = LLMClient(
            LLMConfig(
                provider=run.provider,
                model=run.model,
                temperature=run.temperature,
                seed=run.seed,
                cache=run.cache,
            )
        )
        # Optionally debias the topic before convening (see framing.py). The
        # sensitivity orchestrator sets neutralize=False because it controls the
        # exact wording of each variant itself.
        self.neutralization = None
        if run.neutralize:
            from framing import neutralize_topic

            self._progress(0.02, "Neutralising the topic…")
            self.neutralization = neutralize_topic(self.client, run.topic)
            if self.neutralization.neutral_topic != run.topic:
                print("Topic neutralised for debate:")
                print(f"  original: {run.topic}")
                print(f"  neutral : {self.neutralization.neutral_topic}")
                if self.neutralization.detected_issues:
                    issues = ", ".join(
                        f"{i.get('span','?')} ({i.get('type','?')})"
                        for i in self.neutralization.detected_issues
                    )
                    print(f"  issues  : {issues}")
                self.run.topic = self.neutralization.neutral_topic

        self.personas = load_personas(run.tags, run.require_all)
        if not self.personas:
            raise SystemExit(
                "No council members matched the requested tags. "
                "Try different --tags or drop the filter."
            )
        self.panel = JudgePanel(load_judge_instructions(), self.client, run.judges)
        self.rubric = build_rubric(run.weights)

        # Load persona knowledge cards (compact always-on flavour), if present.
        self._cards: dict[str, str] = {}
        if run.cards:
            for p in self.personas:
                txt = format_card(load_card(p.name))
                if txt:
                    self._cards[p.name] = txt
            if self._cards:
                print(f"Knowledge cards: applied to {len(self._cards)} members")
        print(
            f"Convened {len(self.personas)} members and a "
            f"{len(self.panel.judges)}-judge panel ({', '.join(self.panel.names)}) "
            f"on '{run.provider}:{self.client.model}'"
            + (f" [seed={run.seed}]" if run.seed is not None else "")
        )
        rubric_str = ", ".join(
            f"{a.name}×{a.weight:g}" if a.weight != 1 else a.name for a in self.rubric
        )
        print(f"Idea rubric: {rubric_str}")
        print("Members: " + ", ".join(p.name for p in self.personas))

        # Build a retrieval index over each member's Files/ corpus, once, up
        # front (embeddings are cached, so re-runs are cheap). Parallelised.
        self.retrievers: dict[str, object] = {}
        if run.rag:
            from rag import build_index

            self._progress(0.05, "Indexing member corpora (RAG)…")

            def _index(persona: Persona):
                idx = build_index(
                    os.path.join(MEMBERS_DIR, persona.name), self.client, persona.name
                )
                return persona.name, idx

            for name, idx in pmap(_index, self.personas, run.concurrency):
                if idx is not None:
                    self.retrievers[name] = idx
            if self.retrievers:
                total_chunks = sum(len(r.chunks) for r in self.retrievers.values())
                print(
                    f"RAG: indexed {len(self.retrievers)} member corpora "
                    f"({total_chunks} chunks)"
                )
        self._progress(0.10, f"Convened {len(self.personas)} members")

    def _progress(self, frac: float, msg: str) -> None:
        self._progress_cb({"type": "progress", "frac": max(0.0, min(frac, 0.99)), "msg": msg})

    def _make_member(self, persona: Persona, side: Side) -> "Member":
        return Member(
            persona,
            side,
            self.client,
            retriever=self.retrievers.get(persona.name),
            rag_k=self.run.rag_k,
            card_text=self._cards.get(persona.name, ""),
        )

    # -- idea generation --------------------------------------------------- #
    def gather_ideas(self) -> list[dict]:
        n = len(self.personas)
        done = itertools.count(1)  # count().__next__ is atomic in CPython

        def one(persona: Persona) -> dict:
            member = self._make_member(persona, Side.FOR)
            idea = member.generate_idea(self.run.topic)
            i = next(done)
            self._progress(0.10 + 0.25 * i / n, f"Gathering ideas… {i}/{n}")
            return {"author": persona.name, "idea": idea}

        ideas = pmap(one, self.personas, self.run.concurrency)
        for item in ideas:
            print(f"\n[{item['author']}] proposes:\n{item['idea']}")
        return ideas

    def dedupe_ideas(self, ideas: list[dict]) -> tuple[list[dict], dict]:
        """Drop near-duplicate ideas so the debate budget is spent on distinct
        ones. Returns (unique_ideas, info)."""
        from similarity import dedupe

        texts = [i["idea"] for i in ideas]
        kept, clusters, method = dedupe(texts, self.client, self.run.dedupe_threshold)
        unique = [ideas[i] for i in kept]
        merged = {
            ideas[rep]["author"]: [ideas[d]["author"] for d in dups]
            for rep, dups in clusters.items()
            if dups
        }
        info = {
            "method": method,
            "raw_count": len(ideas),
            "unique_count": len(unique),
            "threshold": self.run.dedupe_threshold,
            "merged": merged,
        }
        if len(unique) < len(ideas):
            print(
                f"\nDe-duplicated ideas: {len(ideas)} -> {len(unique)} distinct "
                f"({method})."
            )
        return unique, info

    # -- debate & scoring -------------------------------------------------- #
    def evaluate_idea(self, idea: str, pos: tuple[int, int] = (0, 1)) -> dict:
        # Map this idea's work into the [0.38, 0.98] progress band.
        j, m = pos
        base, span = 0.38 + 0.60 * (j / m), 0.60 / m
        p = len(self.personas)
        done = itertools.count(1)

        # Phase 1: a structured debate per member (independent across members ->
        # fan out): opening statements, then `rounds` cross-examination rounds,
        # then optional closing statements. Each member keeps its own memory, so
        # later turns build on earlier ones.
        topic = self.run.topic
        rounds = max(1, self.run.rounds)

        def debate(persona: Persona) -> list[dict]:
            pro = self._make_member(persona, Side.FOR)
            con = self._make_member(persona, Side.AGAINST)
            out: list[dict] = []

            def add(label: str, side: str, text: str) -> None:
                out.append({"member": persona.name, "type": label, "side": side, "argument": text})

            last_for = pro.argue(topic, idea)
            last_against = con.argue(topic, idea)
            add("opening", "for", last_for)
            add("opening", "against", last_against)

            for r in range(1, rounds + 1):
                pro_r = pro.rebut(topic, idea, last_against)
                con_r = con.rebut(topic, idea, last_for)
                add(f"rebuttal_{r}", "for", pro_r)
                add(f"rebuttal_{r}", "against", con_r)
                last_for, last_against = pro_r, con_r

            if self.run.closing:
                add("closing", "for", pro.closing(topic, idea))
                add("closing", "against", con.closing(topic, idea))

            k = next(done)
            self._progress(base + span * 0.7 * k / p,
                           f"Idea {j + 1}/{m}: debated {k}/{p} members")
            return out

        per_member = pmap(debate, self.personas, self.run.concurrency)
        entries = [e for sub in per_member for e in sub]

        # Phase 2: score every argument (independent) -> fan out.
        self._progress(base + span * 0.8, f"Idea {j + 1}/{m}: scoring arguments…")

        def score(entry: dict) -> dict:
            verdict = self.panel.score_argument(self.run.topic, idea, entry["argument"])
            return {**entry, "score": verdict["aggregate"], "judge_votes": verdict["votes"]}

        transcript = pmap(score, entries, self.run.concurrency)

        for_total = sum(e["score"] for e in transcript if e["side"] == "for")
        against_total = sum(e["score"] for e in transcript if e["side"] == "against")

        # Phase 3: direct verdict on the idea's own merit, judged from the
        # strongest points each side actually made (bounded to the top few).
        for_case = self._best_points(transcript, "for")
        against_case = self._best_points(transcript, "against")
        idea_verdict = self.panel.score_idea(
            self.run.topic, idea, for_case, against_case, self.rubric
        )
        axes_str = " ".join(f"{k}={v:.0f}" for k, v in idea_verdict["axes"].items())
        self._progress(base + span * 0.98,
                       f"Idea {j + 1}/{m}: verdict {idea_verdict['overall']:.1f}/10")
        print(
            f"  for={for_total:.0f} against={against_total:.0f} "
            f"-> verdict {idea_verdict['overall']:.1f}/10  [{axes_str}]"
        )

        return {
            "idea": idea,
            "verdict_score": idea_verdict["overall"],
            "verdict_axes": idea_verdict["axes"],
            "verdict_votes": idea_verdict["votes"],
            "for_score": for_total,
            "against_score": against_total,
            "total_score": for_total + against_total,
            "margin": for_total - against_total,
            "transcript": transcript,
        }

    @staticmethod
    def _best_points(transcript: list[dict], side: str, limit: int = _TOP_ARGS_PER_SIDE) -> str:
        entries = sorted(
            (e for e in transcript if e["side"] == side),
            key=lambda e: e["score"],
            reverse=True,
        )[:limit]
        if not entries:
            return "(none)"
        return "\n".join(
            f"- [{e['member']}, score {e['score']:.0f}] {e['argument'][:800]}"
            for e in entries
        )

    def run_session(self, write: bool = True) -> dict:
        started = time.time()
        all_ideas = self.gather_ideas()

        # Drop near-duplicates so the debate budget goes to distinct ideas.
        dedupe_info = None
        candidates = all_ideas
        if self.run.dedupe and len(all_ideas) > 1:
            self._progress(0.36, "De-duplicating ideas…")
            candidates, dedupe_info = self.dedupe_ideas(all_ideas)

        # Evaluate a subset (deterministic: the first N distinct ideas).
        n = min(self.run.ideas_to_evaluate, len(candidates))
        selected = candidates[:n]
        print(f"\nEvaluating {n} of {len(candidates)} distinct ideas...\n")

        evaluations = []
        for j, item in enumerate(selected):
            print(f"=== Debating idea by {item['author']} ===")
            evaluations.append(self.evaluate_idea(item["idea"], pos=(j, n)))

        # Best idea = highest direct verdict on the idea's own merit; the total
        # argument quality is a tie-breaker (how much substance the debate had).
        best = (
            max(evaluations, key=lambda e: (e["verdict_score"], e["total_score"]))
            if evaluations
            else None
        )

        result = {
            "topic": self.run.topic,
            "neutralization": self.neutralization.to_dict() if self.neutralization else None,
            "provider": self.run.provider,
            "model": self.client.model,
            "seed": self.run.seed,
            "temperature": self.run.temperature,
            "members": [p.name for p in self.personas],
            "judges": self.panel.names,
            "debate": {"rounds": self.run.rounds, "closing": self.run.closing},
            "cards_used": sorted(self._cards),
            "rubric": [{"axis": a.name, "weight": a.weight} for a in self.rubric],
            "rag": {
                "enabled": self.run.rag,
                "k": self.run.rag_k,
                "indexed_members": {
                    name: len(r.chunks) for name, r in self.retrievers.items()
                },
            },
            "all_ideas": all_ideas,
            "dedupe": dedupe_info,
            "evaluations": evaluations,
            "best_idea": best,
            "usage": self.client.usage_summary(),
            "elapsed_seconds": round(time.time() - started, 1),
        }

        if write:
            with open(self.run.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        if best:
            print(f"BEST IDEA (verdict {best['verdict_score']:.1f}/10):")
            print(best["idea"])
            axes_str = ", ".join(f"{k} {v:.0f}" for k, v in best["verdict_axes"].items())
            print(
                f"\n[verdict {best['verdict_score']:.1f}/10 | {axes_str} | "
                f"debate substance: for {best['for_score']:.0f}, "
                f"against {best['against_score']:.0f}]"
            )
        u = result["usage"]
        cost = u["estimated_cost_usd"]
        cost_str = f"~${cost:.4f}" if cost else "n/a (unpriced model)"
        print(
            f"\nUsage: {u['requests']} requests, {u['cache_hits']} cache hits, "
            f"{u['total_tokens']} tokens, est. cost {cost_str}  "
            f"(in {result['elapsed_seconds']}s)"
        )
        if write:
            print(f"Full results written to {self.run.output}")
        print("=" * 60)
        return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_weights(spec: str) -> dict:
    """Parse 'novelty=2,feasibility=1.5' into {'novelty': 2.0, 'feasibility': 1.5}."""
    weights: dict[str, float] = {}
    valid = {a.name for a in DEFAULT_IDEA_RUBRIC}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("=")
        key = key.strip()
        if key not in valid:
            raise SystemExit(
                f"Unknown rubric axis '{key}'. Valid axes: {', '.join(sorted(valid))}."
            )
        try:
            weights[key] = float(val)
        except ValueError:
            raise SystemExit(f"Weight for '{key}' must be a number, got '{val}'.")
    return weights


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Run the AI Council on a topic.")
    p.add_argument("--topic", help="The question/topic to generate ideas for.")
    p.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter", "openai"],
        help="LLM provider (default: openrouter).",
    )
    p.add_argument("--model", default=None, help="Override the model id.")
    p.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Filter members by tags (e.g. economics innovation).",
    )
    p.add_argument(
        "--require-all-tags",
        action="store_true",
        help="Require members to have ALL given tags (default: any).",
    )
    p.add_argument(
        "--ideas", type=int, default=3, help="How many ideas to debate."
    )
    p.add_argument(
        "--judges",
        type=int,
        default=3,
        help="Size of the judging panel (default: 3; scores are the median).",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Cross-examination rounds after the opening statements (default: 1).",
    )
    p.add_argument(
        "--no-closing",
        dest="closing",
        action="store_false",
        help="Skip closing statements.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility (recorded and sent to the API).",
    )
    p.add_argument(
        "--no-neutralize",
        dest="neutralize",
        action="store_false",
        help="Debate the topic exactly as given (skip the debiasing rewrite).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max parallel LLM calls (default: 8; 1 = serial).",
    )
    p.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help="Debate all generated ideas without dropping near-duplicates.",
    )
    p.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.85,
        help="Similarity above which two ideas are treated as duplicates.",
    )
    p.add_argument(
        "--no-cache",
        dest="cache",
        action="store_false",
        help="Disable the on-disk response cache.",
    )
    p.add_argument(
        "--no-rag",
        dest="rag",
        action="store_false",
        help="Skip retrieval over member Files/ corpora.",
    )
    p.add_argument(
        "--rag-k",
        type=int,
        default=4,
        help="Passages retrieved from a member's corpus per prompt (default: 4).",
    )
    p.add_argument(
        "--no-cards",
        dest="cards",
        action="store_false",
        help="Skip persona knowledge cards (card.json).",
    )
    p.add_argument(
        "--weights",
        default="",
        help="Re-weight idea rubric axes, e.g. 'novelty=2,feasibility=1.5'. "
        "Axes: " + ", ".join(a.name for a in DEFAULT_IDEA_RUBRIC) + ".",
    )
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--output", default="results.json")
    args = p.parse_args()

    topic = args.topic or input("Topic of interest: ").strip()
    return RunConfig(
        weights=parse_weights(args.weights),
        topic=topic,
        tags=args.tags,
        require_all=args.require_all_tags,
        ideas_to_evaluate=args.ideas,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        judges=args.judges,
        rounds=args.rounds,
        closing=args.closing,
        seed=args.seed,
        neutralize=args.neutralize,
        concurrency=args.concurrency,
        dedupe=args.dedupe,
        dedupe_threshold=args.dedupe_threshold,
        cache=args.cache,
        rag=args.rag,
        rag_k=args.rag_k,
        cards=args.cards,
        output=args.output,
    )


if __name__ == "__main__":
    Council(parse_args()).run_session()

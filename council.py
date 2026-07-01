"""AI Council - idea generation, debate and evaluation.

A modern, provider-agnostic rewrite of the original ``openaicouncil.py``.

Key differences from the original:
  * Uses Chat Completions (works with OpenRouter *and* OpenAI) instead of the
    now-deprecated OpenAI Assistants API.
  * Judges return a structured 1-10 rubric score (logic / evidence / rigour)
    instead of the crude VADER sentiment proxy.
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
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum

from llm import LLMClient, LLMConfig


MEMBERS_DIR = "CouncilMembers"


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
# Debating members
# --------------------------------------------------------------------------- #
class Member:
    """A persona that argues one side and remembers its own contributions."""

    def __init__(self, persona: Persona, side: Side, client: LLMClient):
        self.persona = persona
        self.side = side
        self.client = client
        self.history: list[dict] = [{"role": "system", "content": persona.instructions}]

    def _ask(self, prompt: str) -> str:
        self.history.append({"role": "user", "content": prompt})
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
        return self._ask(prompt)

    def argue(self, topic: str, idea: str) -> str:
        stance = "argue in favour of" if self.side is Side.FOR else "argue against"
        prompt = (
            f"Topic: {topic}\nIdea under debate: {idea}\n\n"
            f"Using your expertise, {stance} this idea. Be concise, concrete "
            "and persuasive. Lead with your strongest point."
        )
        return self._ask(prompt)

    def rebut(self, topic: str, idea: str, opponent_arg: str) -> str:
        prompt = (
            f"Topic: {topic}\nIdea under debate: {idea}\n\n"
            f"Your opponent argued:\n\"{opponent_arg}\"\n\n"
            "Write a sharp, concise rebuttal from your side that exposes the "
            "weaknesses in their reasoning while reinforcing your position."
        )
        return self._ask(prompt)


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)


class Judge:
    """Scores arguments on a numeric rubric instead of sentiment polarity."""

    def __init__(self, instructions: str, client: LLMClient):
        self.client = client
        self.instructions = (
            instructions
            + "\n\nYou MUST reply with a JSON object of the form "
            '{"score": <integer 1-10>, "reasoning": "<one or two sentences>"} '
            "where 10 means a flawless, rigorous, well-evidenced argument and 1 "
            "means fallacious or baseless. Reply with JSON only."
        )

    def score(self, topic: str, idea: str, argument: str) -> tuple[float, str]:
        messages = [
            {"role": "system", "content": self.instructions},
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\nIdea: {idea}\n\n"
                    f"Argument to judge:\n\"{argument}\"\n\n"
                    "Score its logical strength, evidence and rigour."
                ),
            },
        ]
        raw = self.client.chat(
            messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> tuple[float, str]:
        try:
            data = json.loads(raw)
            return float(data["score"]), str(data.get("reasoning", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            match = _SCORE_RE.search(raw)
            if match:
                return float(match.group(1)), raw.strip()
            # Neutral fallback so one malformed reply can't crash a whole run.
            return 5.0, raw.strip()


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
    output: str = "results.json"


class Council:
    def __init__(self, run: RunConfig):
        self.run = run
        self.client = LLMClient(
            LLMConfig(
                provider=run.provider,
                model=run.model,
                temperature=run.temperature,
            )
        )
        self.personas = load_personas(run.tags, run.require_all)
        if not self.personas:
            raise SystemExit(
                "No council members matched the requested tags. "
                "Try different --tags or drop the filter."
            )
        self.judge = Judge(load_judge_instructions(), self.client)
        print(
            f"Convened {len(self.personas)} members on "
            f"'{run.provider}:{self.client.model}': "
            + ", ".join(p.name for p in self.personas)
        )

    # -- idea generation --------------------------------------------------- #
    def gather_ideas(self) -> list[dict]:
        ideas = []
        for persona in self.personas:
            member = Member(persona, Side.FOR, self.client)
            idea = member.generate_idea(self.run.topic)
            print(f"\n[{persona.name}] proposes:\n{idea}")
            ideas.append({"author": persona.name, "idea": idea})
        return ideas

    # -- debate & scoring -------------------------------------------------- #
    def evaluate_idea(self, idea: str) -> dict:
        for_total = 0.0
        against_total = 0.0
        transcript = []
        for persona in self.personas:
            pro = Member(persona, Side.FOR, self.client)
            con = Member(persona, Side.AGAINST, self.client)

            pro_arg = pro.argue(self.run.topic, idea)
            con_arg = con.argue(self.run.topic, idea)
            pro_rebut = pro.rebut(self.run.topic, idea, con_arg)
            con_rebut = con.rebut(self.run.topic, idea, pro_arg)

            for label, arg, bucket in [
                ("for_arg", pro_arg, "for"),
                ("against_arg", con_arg, "against"),
                ("for_rebuttal", pro_rebut, "for"),
                ("against_rebuttal", con_rebut, "against"),
            ]:
                score, reasoning = self.judge.score(self.run.topic, idea, arg)
                if bucket == "for":
                    for_total += score
                else:
                    against_total += score
                transcript.append(
                    {
                        "member": persona.name,
                        "type": label,
                        "argument": arg,
                        "score": score,
                        "judge_reasoning": reasoning,
                    }
                )
            print(
                f"  {persona.name}: for={for_total:.0f} against={against_total:.0f}"
            )

        return {
            "idea": idea,
            "for_score": for_total,
            "against_score": against_total,
            "total_score": for_total + against_total,
            "margin": for_total - against_total,
            "transcript": transcript,
        }

    def run_session(self) -> dict:
        started = time.time()
        all_ideas = self.gather_ideas()

        # Evaluate a subset (deterministic: the first N gathered).
        n = min(self.run.ideas_to_evaluate, len(all_ideas))
        selected = all_ideas[:n]
        print(f"\nEvaluating {n} of {len(all_ideas)} ideas...\n")

        evaluations = []
        for item in selected:
            print(f"=== Debating idea by {item['author']} ===")
            evaluations.append(self.evaluate_idea(item["idea"]))

        # Best idea = highest combined argument quality (a proxy for how much
        # substantive debate the idea can sustain).
        best = max(evaluations, key=lambda e: e["total_score"]) if evaluations else None

        result = {
            "topic": self.run.topic,
            "provider": self.run.provider,
            "model": self.client.model,
            "members": [p.name for p in self.personas],
            "all_ideas": all_ideas,
            "evaluations": evaluations,
            "best_idea": best,
            "elapsed_seconds": round(time.time() - started, 1),
        }

        with open(self.run.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        if best:
            print("BEST IDEA (highest sustained debate quality):")
            print(best["idea"])
            print(
                f"\n[for {best['for_score']:.0f} | against "
                f"{best['against_score']:.0f} | total {best['total_score']:.0f}]"
            )
        print(f"\nFull results written to {self.run.output}")
        print("=" * 60)
        return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
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
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--output", default="results.json")
    args = p.parse_args()

    topic = args.topic or input("Topic of interest: ").strip()
    return RunConfig(
        topic=topic,
        tags=args.tags,
        require_all=args.require_all_tags,
        ideas_to_evaluate=args.ideas,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        output=args.output,
    )


if __name__ == "__main__":
    Council(parse_args()).run_session()

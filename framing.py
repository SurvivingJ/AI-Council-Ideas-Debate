"""Topic framing utilities: neutralise a question and generate wording variants.

The topic is the single input that seeds an entire council run, and question
wording carries huge, often invisible influence (loaded terms, presuppositions,
gain/loss framing). This module provides two pre-processing steps:

  * ``neutralize_topic`` - rewrite a topic as a neutral, open question that does
    not presuppose an answer, with a verification loop that checks the rewrite is
    actually neutral and faithful to the original intent.
  * ``generate_variants`` - produce two *kinds* of reworded variants:
      - PARAPHRASES: meaning-preserving rewordings, to test *lexical robustness*
        (if conclusions change here, that is instability/noise - bad).
      - REFRAMES: deliberately re-slanted versions, to test *framing sensitivity*
        (if conclusions change here, that is a genuine finding, not a bug).

Both reuse the provider-agnostic ``LLMClient`` from ``llm.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from llm import LLMClient

# Rewrites scoring below this neutrality (0-1) trigger one corrective retry.
_NEUTRALITY_THRESHOLD = 0.8

_JSON = {"type": "json_object"}


def _chat_json(client: LLMClient, system: str, user: str, temperature: float = 0.4) -> dict:
    raw = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        response_format=_JSON,
    )
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


# --------------------------------------------------------------------------- #
# Neutralisation
# --------------------------------------------------------------------------- #
@dataclass
class Neutralization:
    original_topic: str
    neutral_topic: str
    detected_issues: list[dict]
    presuppositions: list[str]
    neutrality_score: float
    meaning_preserved: bool

    def to_dict(self) -> dict:
        return {
            "original_topic": self.original_topic,
            "neutral_topic": self.neutral_topic,
            "detected_issues": self.detected_issues,
            "presuppositions": self.presuppositions,
            "neutrality_score": self.neutrality_score,
            "meaning_preserved": self.meaning_preserved,
        }


_NEUTRALIZE_SYSTEM = (
    "You are a careful, impartial editor who removes bias from questions before "
    "they are debated. You never answer the question or take a side."
)


def _neutralize_prompt(topic: str, feedback: str = "") -> str:
    extra = f"\n\nA previous attempt was judged not neutral enough: {feedback}\n" if feedback else ""
    return (
        "Rewrite the following topic as a NEUTRAL, open question suitable for an "
        "unbiased debate. Remove loaded or emotive language, remove "
        "presuppositions that assume a particular answer or that something is "
        "inherently good or bad, and avoid biasing toward any side - while "
        "PRESERVING the author's actual intent, scope and specificity. Do not "
        "make it vague or generic." + extra + "\n\nReturn a JSON object: "
        '{"neutral_topic": string, "detected_issues": [{"type": string, '
        '"span": string, "note": string}], "presuppositions": [string]}.'
        f"\n\nTopic: {topic}"
    )


def _verify_neutrality(client: LLMClient, original: str, rewrite: str) -> dict:
    return _chat_json(
        client,
        "You assess whether a rewritten question is neutral and faithful to the "
        "original. You do not answer the question.",
        (
            f"Original: {original}\nRewrite: {rewrite}\n\n"
            "Rate how NEUTRAL and unbiased the rewrite is (0.0 = heavily "
            "loaded/leading, 1.0 = perfectly neutral) and whether it preserves "
            "the original's meaning and scope. Return JSON "
            '{"neutral_score": number, "meaning_preserved": boolean, '
            '"note": string}.'
        ),
        temperature=0.1,
    )


def neutralize_topic(client: LLMClient, topic: str) -> Neutralization:
    """Return a neutralised version of ``topic`` with detected issues.

    Runs one corrective retry if the first rewrite is judged insufficiently
    neutral. Degrades to the original topic if the model returns nothing usable.
    """
    data = _chat_json(client, _NEUTRALIZE_SYSTEM, _neutralize_prompt(topic))
    neutral = str(data.get("neutral_topic") or topic).strip()

    check = _verify_neutrality(client, topic, neutral)
    score = float(check.get("neutral_score", 1.0) or 0.0)
    preserved = bool(check.get("meaning_preserved", True))

    if score < _NEUTRALITY_THRESHOLD or not preserved:
        retry = _chat_json(
            client,
            _NEUTRALIZE_SYSTEM,
            _neutralize_prompt(topic, feedback=str(check.get("note", ""))),
        )
        retry_neutral = str(retry.get("neutral_topic") or neutral).strip()
        recheck = _verify_neutrality(client, topic, retry_neutral)
        # Keep whichever attempt scored better.
        if float(recheck.get("neutral_score", 0.0) or 0.0) >= score:
            data, neutral = retry, retry_neutral
            score = float(recheck.get("neutral_score", score) or score)
            preserved = bool(recheck.get("meaning_preserved", preserved))

    return Neutralization(
        original_topic=topic,
        neutral_topic=neutral,
        detected_issues=list(data.get("detected_issues", []) or []),
        presuppositions=list(data.get("presuppositions", []) or []),
        neutrality_score=round(score, 2),
        meaning_preserved=preserved,
    )


# --------------------------------------------------------------------------- #
# Wording variants
# --------------------------------------------------------------------------- #
@dataclass
class Variant:
    text: str
    kind: str  # "baseline" | "paraphrase" | "reframe"
    note: str = ""

    def to_dict(self) -> dict:
        return {"text": self.text, "kind": self.kind, "note": self.note}


def generate_variants(
    client: LLMClient, topic: str, n_paraphrase: int = 2, n_reframe: int = 2
) -> list[Variant]:
    """Generate paraphrase (meaning-preserving) and reframe (re-slanted) variants.

    The paraphrases test whether conclusions are stable under mere rewording;
    the reframes test how sensitive conclusions are to deliberate framing.
    """
    data = _chat_json(
        client,
        "You generate reworded versions of a question for a robustness test. "
        "You never answer the question.",
        (
            f"Question: {topic}\n\n"
            f"Produce two kinds of rewordings.\n"
            f"1) PARAPHRASES: {n_paraphrase} versions that preserve the meaning "
            "exactly but use different wording and sentence structure (to test "
            "whether conclusions are stable under mere rewording).\n"
            f"2) REFRAMES: {n_reframe} versions that deliberately reframe the "
            "question with a different slant or presupposition - for example a "
            "positive/gain frame versus a negative/loss frame, or assuming a "
            "different default - to test how sensitive conclusions are to "
            "framing. Each reframe must stay on the same underlying subject.\n\n"
            'Return JSON {"paraphrases": [string], "reframes": [{"text": '
            'string, "frame": string}]}.'
        ),
    )

    variants: list[Variant] = []
    for p in list(data.get("paraphrases", []) or [])[:n_paraphrase]:
        if isinstance(p, str) and p.strip():
            variants.append(Variant(text=p.strip(), kind="paraphrase"))
    for r in list(data.get("reframes", []) or [])[:n_reframe]:
        if isinstance(r, dict) and str(r.get("text", "")).strip():
            variants.append(
                Variant(text=r["text"].strip(), kind="reframe", note=str(r.get("frame", "")))
            )
    return variants

"""Build richer council-member personalities by interviewing them.

Instead of hand-writing a persona, this tool conducts a structured interview:
an interviewer LLM asks probing questions about the figure's worldview,
methods, influences, disagreements and rhetorical style; the model answers
*in character*; and the transcript is distilled into a polished
``instructions.txt`` plus a suggested ``info.txt`` tag line.

The result is a persona grounded in specific commitments and turns of phrase,
which produces noticeably more distinctive debate behaviour than a one-line
"You are X" prompt.

Usage
-----
    python interview.py "Joseph Schumpeter"
    python interview.py "Elinor Ostrom" --seed "Nobel economist, commons governance"
    python interview.py "Ada Lovelace" --questions 8 --provider openai

Add ``--write`` to save the generated files into CouncilMembers/<Name>/.
Without it the interview and drafts are printed and saved to a scratch file.
"""

from __future__ import annotations

import argparse
import json
import os
import re

from llm import LLMClient, LLMConfig

MEMBERS_DIR = "CouncilMembers"


def folder_name(display_name: str) -> str:
    """'Joseph Schumpeter' -> 'JosephSchumpeter'."""
    return re.sub(r"[^A-Za-z0-9]", "", display_name.title().replace(" ", ""))


def generate_questions(client: LLMClient, name: str, seed: str, n: int) -> list[str]:
    seed_line = f" Context: {seed}." if seed else ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a world-class interviewer preparing to profile a "
                "thinker for an intellectual debate panel."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Draft {n} incisive interview questions for {name}.{seed_line} "
                "The questions should surface their core convictions, analytical "
                "method, key influences and rivals, characteristic vocabulary, "
                "and how they reason about novel problems. Return a JSON object "
                '{"questions": ["...", ...]} and nothing else.'
            ),
        },
    ]
    raw = client.chat(messages, temperature=0.7, response_format={"type": "json_object"})
    try:
        return list(json.loads(raw)["questions"])[:n]
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fall back to line-splitting if the model ignored the schema.
        return [q.strip("-* ") for q in raw.splitlines() if q.strip()][:n]


def interview(client: LLMClient, name: str, seed: str, questions: list[str]) -> list[dict]:
    seed_line = f" ({seed})" if seed else ""
    history = [
        {
            "role": "system",
            "content": (
                f"You ARE {name}{seed_line}. Answer every question in the first "
                "person, in your authentic voice, drawing on your real ideas, "
                "writings and worldview. Be specific and vivid; use the "
                "vocabulary and rhetorical habits you are known for. 3-6 "
                "sentences per answer."
            ),
        }
    ]
    transcript = []
    for i, q in enumerate(questions, 1):
        history.append({"role": "user", "content": q})
        answer = client.chat(history, temperature=0.85)
        history.append({"role": "assistant", "content": answer})
        transcript.append({"q": q, "a": answer})
        print(f"\nQ{i}. {q}\n> {answer}")
    return transcript


def synthesize_persona(client: LLMClient, name: str, transcript: list[dict]) -> dict:
    convo = "\n\n".join(f"Q: {t['q']}\nA: {t['a']}" for t in transcript)
    messages = [
        {
            "role": "system",
            "content": (
                "You distil interviews into reusable system prompts for "
                "role-played debate agents."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Below is an interview with {name}. Write a persona brief that "
                "can be used as a system prompt so a model can convincingly "
                f"argue as {name} in a debate.\n\n"
                "Return a JSON object with keys:\n"
                '  "instructions": a rich, second-person persona brief with two '
                "labelled sections, 'Background Info:' and 'Prompting:', "
                "capturing their worldview, method, signature ideas and voice.\n"
                '  "tags": a short comma-separated string of 2-5 lowercase '
                "topic tags (e.g. 'economics,innovation').\n\n"
                f"Interview:\n{convo}"
            ),
        },
    ]
    raw = client.chat(messages, temperature=0.5, response_format={"type": "json_object"})
    try:
        data = json.loads(raw)
        return {
            "instructions": data["instructions"].strip(),
            "tags": data.get("tags", "").strip(),
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"instructions": raw.strip(), "tags": ""}


def save_persona(name: str, persona: dict) -> str:
    folder = os.path.join(MEMBERS_DIR, folder_name(name))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "instructions.txt"), "w", encoding="utf-8") as f:
        f.write(persona["instructions"] + "\n")
    if persona["tags"]:
        with open(os.path.join(folder, "info.txt"), "w", encoding="utf-8") as f:
            f.write(persona["tags"])
    return folder


def main() -> None:
    ap = argparse.ArgumentParser(description="Interview a figure to build a council persona.")
    ap.add_argument("name", help="Display name of the figure, e.g. 'Joseph Schumpeter'.")
    ap.add_argument("--seed", default="", help="Optional one-line context/description.")
    ap.add_argument("--questions", type=int, default=6, help="Number of interview questions.")
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write files into CouncilMembers/<Name>/ (otherwise print only).",
    )
    args = ap.parse_args()

    client = LLMClient(LLMConfig(provider=args.provider, model=args.model))
    print(f"Interviewing {args.name} via {args.provider}:{client.model}\n")

    questions = generate_questions(client, args.name, args.seed, args.questions)
    transcript = interview(client, args.name, args.seed, questions)
    persona = synthesize_persona(client, args.name, transcript)

    print("\n" + "=" * 60)
    print("GENERATED PERSONA (instructions.txt):\n")
    print(persona["instructions"])
    print(f"\nSuggested tags (info.txt): {persona['tags']}")
    print("=" * 60)

    if args.write:
        folder = save_persona(args.name, persona)
        print(f"\nSaved persona to {folder}/")
    else:
        print("\n(Run again with --write to save into CouncilMembers/.)")


if __name__ == "__main__":
    main()

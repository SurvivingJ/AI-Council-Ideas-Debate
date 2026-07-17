"""Build richer council-member personalities by interviewing them.

Instead of hand-writing a persona, this tool conducts a structured interview:
an interviewer LLM asks probing questions about the figure's worldview,
methods, influences, disagreements and rhetorical style; the model answers
*in character*; and the transcript is distilled into a polished
``instructions.txt`` plus a suggested ``info.txt`` tag line.

The result is a persona grounded in specific commitments and turns of phrase,
which produces noticeably more distinctive debate behaviour than a one-line
"You are X" prompt.

The interview has depth knobs beyond the base questions:
  * ``--followups N`` — after the base round, an interviewer picks the *weakest,
    vaguest* prior answer and probes it for something concrete.
  * ``--adversarial N`` — questions channelling the figure's real critics/rivals
    that the persona must defend against.
These deepen the transcript before it is distilled, yielding a sharper persona.

Usage
-----
    python interview.py "Joseph Schumpeter"
    python interview.py "Elinor Ostrom" --seed "Nobel economist, commons governance"
    python interview.py "Ada Lovelace" --questions 8 --followups 3 --adversarial 3

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


def persona_history(name: str, seed: str) -> list[dict]:
    seed_line = f" ({seed})" if seed else ""
    return [{
        "role": "system",
        "content": (
            f"You ARE {name}{seed_line}. Answer every question in the first "
            "person, in your authentic voice, drawing on your real ideas, "
            "writings and worldview. Be specific and vivid; use the vocabulary "
            "and rhetorical habits you are known for. 3-6 sentences per answer."
        ),
    }]


def ask_persona(client: LLMClient, history: list[dict], question: str,
                transcript: list[dict], kind: str) -> str:
    """Ask the in-character persona a question, keeping conversation memory.

    Streams the answer to the console token-by-token."""
    history.append({"role": "user", "content": question})
    tag = "" if kind == "base" else f" [{kind}]"
    print(f"\nQ{tag}. {question}\n> ", end="", flush=True)
    answer = client.chat_stream(
        history, on_token=lambda d: print(d, end="", flush=True), temperature=0.85
    )
    print()
    history.append({"role": "assistant", "content": answer})
    transcript.append({"q": question, "a": answer, "kind": kind})
    return answer


def consistency_check(client: LLMClient, name: str, instructions: str, n: int) -> dict:
    """Sanity-check a synthesised persona: ask factual/positional questions,
    answer them *as the persona brief*, and have a judge rate how well each
    answer matches the real figure (0-1). Returns an average score + details."""
    raw = client.chat(
        [
            {"role": "system", "content": (
                "You write factual check questions to test whether a persona "
                "matches a real thinker's known views.")},
            {"role": "user", "content": (
                f"Write {n} questions about {name}'s well-established views or "
                'methods. Return JSON {"questions": ["...", ...]}.')},
        ],
        temperature=0.5, response_format={"type": "json_object"},
    )
    try:
        questions = list(json.loads(raw)["questions"])[:n]
    except (json.JSONDecodeError, KeyError, TypeError):
        questions = []

    items = []
    for q in questions:
        answer = client.chat(
            [{"role": "system", "content": instructions},
             {"role": "user", "content": q}],
            temperature=0.4,
        )
        judged = client.chat(
            [
                {"role": "system", "content": (
                    f"You judge whether an answer is consistent with, and "
                    f"accurate for, the real {name}.")},
                {"role": "user", "content": (
                    f"Question: {q}\nAnswer: {answer}\n\nRate 0.0-1.0 how "
                    f"consistent and accurate this is for {name}. Return JSON "
                    '{"score": <0.0-1.0>, "note": "<short>"}.')},
            ],
            temperature=0.1, response_format={"type": "json_object"},
        )
        try:
            d = json.loads(judged)
            score = float(d.get("score", 0.5))
            note = str(d.get("note", "")).strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            score, note = 0.5, ""
        items.append({"q": q, "a": answer, "score": score, "note": note})

    avg = round(sum(i["score"] for i in items) / len(items), 2) if items else 0.0
    return {"score": avg, "items": items}


def _convo(transcript: list[dict]) -> str:
    return "\n\n".join(f"Q: {t['q']}\nA: {t['a']}" for t in transcript)


def weakest_followup(client: LLMClient, name: str, transcript: list[dict]) -> str:
    """Interviewer picks the weakest prior answer and probes it."""
    raw = client.chat(
        [
            {"role": "system", "content": (
                "You are a sharp interviewer who probes weak, vague or evasive "
                "answers until they become concrete.")},
            {"role": "user", "content": (
                f"Interview with {name} so far:\n\n{_convo(transcript)}\n\n"
                "Identify the SINGLE weakest, vaguest or least substantive answer, "
                "and write ONE incisive follow-up question that forces a concrete, "
                'specific response. Return JSON {"followup": "..."}.')},
        ],
        temperature=0.6, response_format={"type": "json_object"},
    )
    try:
        return json.loads(raw)["followup"].strip()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return "Can you be far more concrete and specific about your weakest answer above?"


def adversarial_question(client: LLMClient, name: str, transcript: list[dict]) -> str:
    """Generate a tough objection a real critic/rival would press."""
    raw = client.chat(
        [
            {"role": "system", "content": (
                "You channel a thinker's toughest, best-informed critics and rivals.")},
            {"role": "user", "content": (
                f"Given {name}'s positions so far:\n\n{_convo(transcript)}\n\n"
                "Pose ONE tough, adversarial question that a leading critic or rival "
                "of theirs would press — a real objection they must defend against, "
                'not a softball. Return JSON {"question": "..."}.')},
        ],
        temperature=0.8, response_format={"type": "json_object"},
    )
    try:
        return json.loads(raw)["question"].strip()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return f"What is the strongest objection to your view, and how do you answer it?"


def run_interview(client: LLMClient, name: str, seed: str, questions: list[str],
                  followups: int = 0, adversarial: int = 0) -> list[dict]:
    """Full interview: base questions, then weakest-answer follow-ups, then
    adversarial defences — all in one continuous in-character conversation."""
    history = persona_history(name, seed)
    transcript: list[dict] = []
    for q in questions:
        ask_persona(client, history, q, transcript, "base")
    for _ in range(max(0, followups)):
        ask_persona(client, history, weakest_followup(client, name, transcript),
                    transcript, "follow-up")
    for _ in range(max(0, adversarial)):
        ask_persona(client, history, adversarial_question(client, name, transcript),
                    transcript, "adversarial")
    return transcript


def synthesize_persona(client: LLMClient, name: str, transcript: list[dict]) -> dict:
    convo = "\n\n".join(
        f"Q ({t.get('kind', 'base')}): {t['q']}\nA: {t['a']}" for t in transcript
    )
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
    ap.add_argument("--questions", type=int, default=6, help="Number of base questions.")
    ap.add_argument("--followups", type=int, default=2,
                    help="Follow-up questions probing the weakest prior answers.")
    ap.add_argument("--adversarial", type=int, default=2,
                    help="Adversarial questions the figure must defend against.")
    ap.add_argument("--consistency", type=int, default=0,
                    help="Run N self-consistency check questions on the result.")
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
    transcript = run_interview(client, args.name, args.seed, questions,
                               followups=args.followups, adversarial=args.adversarial)
    persona = synthesize_persona(client, args.name, transcript)

    print("\n" + "=" * 60)
    print("GENERATED PERSONA (instructions.txt):\n")
    print(persona["instructions"])
    print(f"\nSuggested tags (info.txt): {persona['tags']}")
    print("=" * 60)

    if args.consistency > 0:
        print("\nSelf-consistency check…")
        chk = consistency_check(client, args.name, persona["instructions"], args.consistency)
        print(f"Consistency score: {chk['score']:.2f}/1.0 "
              "(1.0 = fully in character and accurate)")
        for it in chk["items"]:
            print(f"  [{it['score']:.2f}] {it['q']}")

    if args.write:
        folder = save_persona(args.name, persona)
        print(f"\nSaved persona to {folder}/")
    else:
        print("\n(Run again with --write to save into CouncilMembers/.)")


if __name__ == "__main__":
    main()

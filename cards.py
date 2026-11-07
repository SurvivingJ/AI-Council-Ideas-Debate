"""Generate persona knowledge cards (card.json) for council members.

A knowledge card is a compact, structured distillation of a figure — signature
concepts, characteristic vocabulary, positions they hold, and ideas/thinkers
they push against. The council injects it into the member's system prompt as
always-on flavour so they argue recognisably like themselves (see
``council.load_card`` / ``format_card``).

Cards contain **no fabricated verbatim quotations** — only real, well-attested
concepts and terms of art.

Usage
-----
    python cards.py "JosephSchumpeter" --write
    python cards.py --all --write                 # every member missing a card
    python cards.py --all --overwrite --write     # regenerate all
"""

from __future__ import annotations

import argparse
import json
import os

from llm import LLMClient, LLMConfig

MEMBERS_DIR = "CouncilMembers"
FIELDS = ["signature_concepts", "key_terms", "stances", "rivals"]


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def member_context(name: str) -> str:
    """Persona instructions plus a short excerpt of any key_ideas notes."""
    folder = os.path.join(MEMBERS_DIR, name)
    ctx = _read(os.path.join(folder, "instructions.txt"))
    notes = _read(os.path.join(folder, "Files", "key_ideas.md"))
    if notes:
        ctx += "\n\nReference notes:\n" + notes[:4000]
    return ctx.strip()


def generate_card(client: LLMClient, name: str) -> dict:
    display = name  # folder names are already the recognisable name (CamelCase)
    raw = client.chat(
        [
            {"role": "system", "content": (
                "You distil a thinker into a compact 'knowledge card' used to keep "
                "a role-played debate agent recognisably in character. Use only "
                "real, well-attested material. Do NOT invent verbatim quotations."
            )},
            {"role": "user", "content": (
                f"Build a knowledge card for {display}, using this context:\n\n"
                f"{member_context(name)}\n\n"
                "Return a JSON object with these keys, each a short list (3-6) of "
                "concise strings:\n"
                '  "signature_concepts": the ideas/theories they are known for;\n'
                '  "key_terms": characteristic words/phrases they actually use '
                "(terms of art, not fabricated quotes);\n"
                '  "stances": positions they characteristically take;\n'
                '  "rivals": ideas, schools or thinkers they push against.\n'
                "Reply with JSON only."
            )},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        data = {}
    # Normalise to lists of clean strings.
    card = {}
    for k in FIELDS:
        vals = data.get(k) or []
        if isinstance(vals, str):
            vals = [vals]
        card[k] = [str(v).strip() for v in vals if str(v).strip()][:6]
    return card


def save_card(name: str, card: dict) -> str:
    path = os.path.join(MEMBERS_DIR, name, "card.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    return path


def member_names() -> list[str]:
    out = []
    for entry in sorted(os.listdir(MEMBERS_DIR)):
        folder = os.path.join(MEMBERS_DIR, entry)
        if os.path.isdir(folder) and entry.lower() != "judge" \
                and os.path.exists(os.path.join(folder, "instructions.txt")):
            out.append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate persona knowledge cards.")
    ap.add_argument("name", nargs="?", help="Member folder name, e.g. JosephSchumpeter.")
    ap.add_argument("--all", action="store_true", help="Process every member.")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate existing cards.")
    ap.add_argument("--write", action="store_true", help="Write card.json (else print).")
    ap.add_argument("--provider", default="openrouter", choices=["openrouter", "openai"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    client = LLMClient(LLMConfig(provider=args.provider, model=args.model))

    if args.all:
        names = member_names()
    elif args.name:
        names = [args.name]
    else:
        ap.error("Provide a member name or --all.")

    for name in names:
        card_path = os.path.join(MEMBERS_DIR, name, "card.json")
        if args.all and not args.overwrite and os.path.exists(card_path):
            continue
        print(f"\n=== {name} ===")
        card = generate_card(client, name)
        print(json.dumps(card, indent=2, ensure_ascii=False))
        if args.write:
            print("wrote", save_card(name, card))


if __name__ == "__main__":
    main()

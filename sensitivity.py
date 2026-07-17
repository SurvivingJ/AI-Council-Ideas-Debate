"""Wording-sensitivity analysis for the AI Council.

Runs the council on several wordings of the same question - a neutral baseline,
meaning-preserving PARAPHRASES, and deliberately re-slanted REFRAMES - holding
the seed fixed so differences are attributable to *wording*, not sampling noise.
It then reports two distinct things:

  * LEXICAL ROBUSTNESS  (baseline vs paraphrases): the answer *should* be stable.
    If it moves here, that is instability/noise - a reason to distrust the result.
  * FRAMING SENSITIVITY  (baseline vs reframes): the answer moving here is a
    genuine finding - the recommendation is an artifact of how the question is
    framed.

Idea overlap is measured with embeddings when available, falling back to a
lexical similarity otherwise.

Usage
-----
    python sensitivity.py --topic "How do we stop kids wasting time online?"
    python sensitivity.py --topic "..." --paraphrases 3 --reframes 3 --ideas 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics

from llm import LLMClient, LLMConfig
from framing import neutralize_topic, generate_variants, Variant
from council import Council, RunConfig
from similarity import similarity_fn


def _sum_usage(usages: list[dict]) -> dict:
    total = {"requests": 0, "cache_hits": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "estimated_cost_usd": 0.0}
    for u in usages:
        for k in total:
            total[k] += u.get(k, 0) or 0
    total["estimated_cost_usd"] = round(total["estimated_cost_usd"], 6)
    total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
    return total


def _label(value: float, high: float, medium: float, invert: bool = False) -> str:
    """Bucket a 0-1 value. invert=True means low similarity -> HIGH (sensitivity)."""
    if invert:
        if value <= (1 - high):
            return "HIGH"
        if value <= (1 - medium):
            return "MEDIUM"
        return "LOW"
    if value >= high:
        return "HIGH"
    if value >= medium:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_sensitivity(
    base: RunConfig, n_paraphrase: int, n_reframe: int, output: str, progress_cb=None
) -> dict:
    emit = progress_cb or (lambda ev: None)
    client = LLMClient(
        LLMConfig(provider=base.provider, model=base.model, seed=base.seed, cache=base.cache)
    )

    emit({"type": "progress", "frac": 0.02, "msg": "Neutralising topic…"})
    print("Neutralising topic...")
    neutral = neutralize_topic(client, base.topic)
    print(f"  baseline (neutral): {neutral.neutral_topic}")

    emit({"type": "progress", "frac": 0.06, "msg": "Generating wording variants…"})
    print("Generating wording variants...")
    variants = generate_variants(client, neutral.neutral_topic, n_paraphrase, n_reframe)
    wordings: list[Variant] = [
        Variant(text=neutral.neutral_topic, kind="baseline", note="neutralised original")
    ] + variants
    for v in wordings:
        print(f"  [{v.kind}] {v.text}" + (f"  ({v.note})" if v.note else ""))

    # Run the council on each wording, seed fixed, neutralisation off (we control
    # the exact wording here).
    runs = []
    usages = []
    total = len(wordings)
    for i, v in enumerate(wordings):
        emit({"type": "progress", "frac": 0.1 + 0.85 * i / total,
              "msg": f"Council {i + 1}/{total} on [{v.kind}]: {v.text[:40]}"})
        print(f"\n{'#' * 60}\n# Running council on [{v.kind}]: {v.text}\n{'#' * 60}")
        cfg = dataclasses.replace(base, topic=v.text, neutralize=False)
        result = Council(cfg).run_session(write=False)
        usages.append(result.get("usage", {}))
        best = result.get("best_idea") or {}
        runs.append(
            {
                "wording": v.text,
                "kind": v.kind,
                "frame": v.note,
                "best_idea": best.get("idea", ""),
                "verdict_score": best.get("verdict_score"),
            }
        )

    emit({"type": "progress", "frac": 0.96, "msg": "Comparing wordings…"})
    report = _compare(runs, client)
    total_usage = _sum_usage([client.usage_summary(), *usages])
    out = {
        "original_topic": base.topic,
        "neutralization": neutral.to_dict(),
        "model": client.model,
        "seed": base.seed,
        "runs": runs,
        "analysis": report,
        "usage": total_usage,
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    _print_report(base.topic, neutral.neutral_topic, runs, report, output)
    cost = total_usage["estimated_cost_usd"]
    print(
        f"Total usage across {len(wordings)} council runs: "
        f"{total_usage['requests']} requests, {total_usage['cache_hits']} cache hits, "
        f"{total_usage['total_tokens']} tokens, est. cost "
        + (f"~${cost:.4f}" if cost else "n/a")
    )
    return out


def _compare(runs: list[dict], client: LLMClient) -> dict:
    winners = [r["best_idea"] or "(none)" for r in runs]
    sim, method = similarity_fn(winners, client)

    # runs[0] is the baseline.
    to_baseline = {i: round(sim(0, i), 3) for i in range(1, len(runs))}

    para_idx = [i for i, r in enumerate(runs) if r["kind"] == "paraphrase"]
    reframe_idx = [i for i, r in enumerate(runs) if r["kind"] == "reframe"]

    para_sims = [sim(0, i) for i in para_idx]
    reframe_sims = [sim(0, i) for i in reframe_idx]

    # Verdict spread across the "should-be-equivalent" group (baseline + paraphrases).
    equiv_verdicts = [
        runs[i]["verdict_score"]
        for i in [0, *para_idx]
        if isinstance(runs[i]["verdict_score"], (int, float))
    ]
    verdict_spread = (
        round(statistics.pstdev(equiv_verdicts), 2) if len(equiv_verdicts) > 1 else 0.0
    )

    lexical_mean = round(statistics.mean(para_sims), 3) if para_sims else None
    framing_mean = round(statistics.mean(reframe_sims), 3) if reframe_sims else None

    return {
        "similarity_method": method,
        "winner_similarity_to_baseline": to_baseline,
        "lexical_robustness": {
            "mean_paraphrase_similarity": lexical_mean,
            "verdict_spread": verdict_spread,
            "label": _label(lexical_mean, 0.8, 0.6) if lexical_mean is not None else "n/a",
        },
        "framing_sensitivity": {
            "mean_reframe_similarity": framing_mean,
            "label": _label(framing_mean, 0.5, 0.25, invert=True)
            if framing_mean is not None
            else "n/a",
        },
    }


def _print_report(
    original: str, neutral: str, runs: list[dict], report: dict, output: str
) -> None:
    lex = report["lexical_robustness"]
    frm = report["framing_sensitivity"]
    print("\n" + "=" * 60)
    print("WORDING-SENSITIVITY REPORT")
    print("=" * 60)
    print(f"Original : {original}")
    print(f"Neutral  : {neutral}")
    print(f"Similarity measured via: {report['similarity_method']}\n")

    for r in runs:
        v = r["verdict_score"]
        vs = f"{v:.1f}/10" if isinstance(v, (int, float)) else "n/a"
        tag = r["kind"] + (f"/{r['frame']}" if r["frame"] else "")
        print(f"[{tag}] verdict {vs}: {r['best_idea'][:70]}")

    print(
        f"\nLEXICAL ROBUSTNESS: {lex['label']}  "
        f"(mean paraphrase winner-similarity="
        f"{lex['mean_paraphrase_similarity']}, verdict spread={lex['verdict_spread']})"
    )
    if lex["label"] == "LOW":
        print("  -> The winner shifts under mere rewording: treat the result as noisy.")
    print(
        f"FRAMING SENSITIVITY: {frm['label']}  "
        f"(mean reframe winner-similarity={frm['mean_reframe_similarity']})"
    )
    if frm["label"] == "HIGH":
        print("  -> The recommendation depends heavily on how the question is framed.")
    print(f"\nFull analysis written to {output}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Wording-sensitivity analysis for the Council.")
    p.add_argument("--topic", help="The question to analyse.")
    p.add_argument("--provider", default="openrouter", choices=["openrouter", "openai"])
    p.add_argument("--model", default=None)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--require-all-tags", action="store_true")
    p.add_argument("--ideas", type=int, default=2, help="Ideas debated per wording.")
    p.add_argument("--judges", type=int, default=3)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed seed (default 0) so differences reflect wording, not noise.",
    )
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--paraphrases", type=int, default=2)
    p.add_argument("--reframes", type=int, default=2)
    p.add_argument("--output", default="sensitivity.json")
    args = p.parse_args()

    topic = args.topic or input("Topic of interest: ").strip()
    base = RunConfig(
        topic=topic,
        tags=args.tags,
        require_all=args.require_all_tags,
        ideas_to_evaluate=args.ideas,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        judges=args.judges,
        seed=args.seed,
        neutralize=False,  # the orchestrator neutralises once, up front
    )
    run_sensitivity(base, args.paraphrases, args.reframes, args.output)


if __name__ == "__main__":
    main()

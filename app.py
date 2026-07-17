"""Streamlit web UI for the AI Council.

Run with:  streamlit run app.py

Configure a run in the sidebar (topic, provider/model, roster tags, judges,
ideas, rubric weights, performance toggles) and launch it, or load an existing
results.json to browse. Renders the neutralised topic, the best idea with its
multi-axis rubric breakdown, an idea-verdict comparison, the full debate
transcript, and the usage/cost summary.
"""

from __future__ import annotations

import json
import os
import queue
import threading

import altair as alt
import pandas as pd
import streamlit as st

from council import Council, RunConfig, DEFAULT_IDEA_RUBRIC, load_personas

# --- Palette (validated defaults from the dataviz reference; single hue for
# single-series magnitude charts, recessive ink for axes/labels). ------------
SERIES = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXES = [a.name for a in DEFAULT_IDEA_RUBRIC]

st.set_page_config(page_title="AI Council", page_icon="🏛️", layout="wide")


# --------------------------------------------------------------------------- #
# Charts (single series -> one hue, no legend; values direct-labelled)
# --------------------------------------------------------------------------- #
def bar_chart(df: pd.DataFrame, cat: str, val: str, title: str, vmax: float | None = None):
    base = alt.Chart(df).encode(
        y=alt.Y(f"{cat}:N", sort=None, title=None,
                axis=alt.Axis(labelColor=INK, domainColor=MUTED, ticks=False)),
        x=alt.X(f"{val}:Q", title=title,
                scale=alt.Scale(domain=[0, vmax]) if vmax else alt.Undefined,
                axis=alt.Axis(labelColor=MUTED, titleColor=MUTED,
                              gridColor=GRID, domainColor=MUTED)),
    )
    bars = base.mark_bar(color=SERIES, cornerRadius=3, height=16)
    labels = base.mark_text(align="left", dx=4, color=INK, fontSize=12).encode(
        text=alt.Text(f"{val}:Q", format=".1f")
    )
    return (bars + labels).properties(height=max(120, 34 * len(df)))


# --------------------------------------------------------------------------- #
# Rendering an existing result dict
# --------------------------------------------------------------------------- #
def render_result(res: dict) -> None:
    st.subheader(res.get("topic", "(no topic)"))

    neu = res.get("neutralization")
    if neu and neu.get("neutral_topic") != neu.get("original_topic"):
        with st.expander("🪄 Topic was neutralised for the debate", expanded=False):
            st.markdown(f"**Original:** {neu['original_topic']}")
            st.markdown(f"**Neutral:** {neu['neutral_topic']}")
            if neu.get("detected_issues"):
                st.caption("Detected framing issues")
                st.dataframe(pd.DataFrame(neu["detected_issues"]),
                             use_container_width=True, hide_index=True)
            if neu.get("presuppositions"):
                st.caption("Presuppositions removed: " + "; ".join(neu["presuppositions"]))

    best = res.get("best_idea")
    usage = res.get("usage", {}) or {}
    cols = st.columns(5)
    cols[0].metric("Best verdict", f"{best['verdict_score']:.1f}/10" if best else "—")
    cols[1].metric("Members", len(res.get("members", [])))
    cols[2].metric("Judges", len(res.get("judges", [])))
    cols[3].metric("Tokens", f"{usage.get('total_tokens', 0):,}")
    cost = usage.get("estimated_cost_usd") or 0
    cols[4].metric("Est. cost", f"${cost:.4f}" if cost else "n/a")

    evaluations = res.get("evaluations", []) or []

    if best:
        st.markdown("### 🏆 Best idea")
        st.success(best["idea"])
        axes = best.get("verdict_axes") or {}
        if axes:
            df = pd.DataFrame({"axis": list(axes.keys()), "score": list(axes.values())})
            st.altair_chart(bar_chart(df, "axis", "score", "score (1–10)", vmax=10),
                            use_container_width=True, theme=None)

    # Idea comparison — single hue; the winner is marked with a star, not recoloured.
    if len(evaluations) > 1:
        st.markdown("### Idea verdicts")
        best_idea_text = best["idea"] if best else None
        rows = []
        for e in evaluations:
            label = e["idea"].split("\n")[0][:48]
            if e["idea"] == best_idea_text:
                label = "★ " + label
            rows.append({"idea": label, "verdict": e["verdict_score"]})
        st.altair_chart(bar_chart(pd.DataFrame(rows), "idea", "verdict", "verdict (1–10)", vmax=10),
                        use_container_width=True, theme=None)

    # Debate transcripts
    if evaluations:
        st.markdown("### Debate transcripts")
        for e in evaluations:
            head = e["idea"].split("\n")[0][:70]
            with st.expander(f"{head}  —  verdict {e['verdict_score']:.1f}/10"):
                axes = e.get("verdict_axes") or {}
                if axes:
                    st.caption(" · ".join(f"{k} {v:.0f}" for k, v in axes.items()))
                tdf = pd.DataFrame([
                    {"member": t["member"], "type": t["type"], "score": t.get("score"),
                     "argument": t["argument"]}
                    for t in e.get("transcript", [])
                ])
                if not tdf.empty:
                    st.dataframe(tdf, use_container_width=True, hide_index=True,
                                 column_config={"argument": st.column_config.TextColumn(width="large")})

    # Ideas & de-duplication
    with st.expander("💡 All generated ideas & de-duplication"):
        ded = res.get("dedupe")
        if ded:
            st.caption(f"{ded['raw_count']} generated → {ded['unique_count']} distinct "
                       f"({ded['method']}).")
            if ded.get("merged"):
                for rep, dups in ded["merged"].items():
                    st.caption(f"• {rep} absorbed duplicates from: {', '.join(dups)}")
        for item in res.get("all_ideas", []):
            st.markdown(f"**{item['author']}** — {item['idea']}")

    # RAG + rubric + raw
    rag = res.get("rag") or {}
    if rag.get("indexed_members"):
        with st.expander("📚 Retrieval corpora (RAG)"):
            st.caption(f"k={rag.get('k')} passages per prompt")
            st.dataframe(pd.DataFrame(
                [{"member": m, "chunks": c} for m, c in rag["indexed_members"].items()]),
                use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download results.json",
                       json.dumps(res, indent=2, ensure_ascii=False),
                       file_name="results.json", mime="application/json")


# --------------------------------------------------------------------------- #
# Rendering a sensitivity-analysis result
# --------------------------------------------------------------------------- #
def render_sensitivity(res: dict) -> None:
    st.subheader("🔬 Wording-sensitivity analysis")
    st.caption(f"Original topic: {res.get('original_topic', '')}")
    neu = res.get("neutralization") or {}
    if neu.get("neutral_topic"):
        st.caption(f"Neutral baseline: {neu['neutral_topic']}")

    a = res.get("analysis", {}) or {}
    lex, frm = a.get("lexical_robustness", {}), a.get("framing_sensitivity", {})
    u = res.get("usage", {}) or {}
    cost = u.get("estimated_cost_usd") or 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Lexical robustness", lex.get("label", "—"),
              help="Winner stability across paraphrases — should be HIGH.")
    c2.metric("Framing sensitivity", frm.get("label", "—"),
              help="How much the winner shifts across deliberate reframes.")
    c3.metric("Est. cost", f"${cost:.4f}" if cost else "n/a")

    if frm.get("label") == "HIGH":
        st.warning("⚠️ The recommendation depends heavily on how the question is framed.")
    if lex.get("label") == "LOW":
        st.warning("⚠️ The winner shifts under mere rewording — treat the result as noisy.")
    st.caption(
        f"Similarity via {a.get('similarity_method', '?')}. Lexical robustness "
        "compares paraphrases (should be stable — instability is noise); framing "
        "sensitivity compares reframes (divergence is a genuine finding)."
    )

    runs = res.get("runs", []) or []
    rows = []
    for r in runs:
        lab = r["kind"] + (f"/{r['frame']}" if r.get("frame") else "")
        rows.append({"wording": f"{lab}: {r['wording'][:38]}",
                     "verdict": r.get("verdict_score") or 0})
    if rows:
        st.altair_chart(bar_chart(pd.DataFrame(rows), "wording", "verdict",
                                  "verdict (1–10)", vmax=10),
                        use_container_width=True, theme=None)

    st.dataframe(pd.DataFrame([
        {"kind": r["kind"], "frame": r.get("frame", ""),
         "verdict": r.get("verdict_score"), "best idea": r.get("best_idea", "")[:90],
         "wording": r["wording"]} for r in runs]),
        use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download sensitivity.json",
                       json.dumps(res, indent=2, ensure_ascii=False),
                       file_name="sensitivity.json", mime="application/json")


# --------------------------------------------------------------------------- #
# Sidebar — configure & launch a run
# --------------------------------------------------------------------------- #
def read_tags() -> list[str]:
    try:
        with open("master_tags.txt", encoding="utf-8") as f:
            return [t.strip() for t in f.read().split(",") if t.strip() and t.strip() != "test"]
    except OSError:
        return []


def sidebar() -> RunConfig | None:
    st.sidebar.title("🏛️ AI Council")

    provider = st.sidebar.selectbox("Provider", ["openrouter", "openai"])
    key_env = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
    if os.environ.get(key_env):
        st.sidebar.caption(f"✅ {key_env} is set")
    else:
        st.sidebar.warning(f"{key_env} not set — a live run will fail.")

    topic = st.sidebar.text_area("Topic / question", height=80,
                                 placeholder="e.g. How to boost local civic participation")
    model = st.sidebar.text_input("Model (blank = provider default)")

    tags = st.sidebar.multiselect("Filter roster by tags", read_tags())
    require_all = st.sidebar.checkbox("Require all tags", value=False)
    try:
        n_members = len(load_personas(tags, require_all))
        st.sidebar.caption(f"{n_members} council members match")
    except Exception:  # noqa: BLE001
        pass

    c1, c2 = st.sidebar.columns(2)
    ideas = c1.slider("Ideas", 1, 10, 3)
    judges = c2.slider("Judges", 1, 5, 3)
    rounds = c1.slider("Debate rounds", 1, 4, 1,
                       help="Cross-examination rounds after the opening statements.")
    rag_k = c2.slider("RAG passages", 0, 10, 4)
    concurrency = c1.slider("Concurrency", 1, 32, 8)
    closing = c2.checkbox("Closing", value=True, help="Include closing statements.")

    use_seed = st.sidebar.checkbox("Fixed seed (reproducible)", value=True)
    seed = st.sidebar.number_input("Seed", value=0, step=1) if use_seed else None

    with st.sidebar.expander("Rubric weights"):
        weights = {ax: st.slider(ax, 0.0, 5.0, 1.0, 0.5) for ax in AXES}
    weights = {k: v for k, v in weights.items() if v != 1.0}

    with st.sidebar.expander("Options"):
        neutralize = st.checkbox("Neutralise topic", value=True)
        dedupe = st.checkbox("De-duplicate ideas", value=True)
        rag = st.checkbox("Retrieval (RAG)", value=True)
        cards = st.checkbox("Knowledge cards", value=True)
        cache = st.checkbox("Use response cache", value=True)

    with st.sidebar.expander("Wording-sensitivity analysis"):
        st.caption("Run the council across several wordings (baseline + "
                   "paraphrases + reframes) to test how framing-dependent the "
                   "answer is. Costs one full run per wording.")
        sensitivity_mode = st.checkbox("Enable sensitivity analysis", value=False)
        paraphrases = st.slider("Paraphrases", 0, 4, 2)
        reframes = st.slider("Reframes", 0, 4, 2)

    label = "▶ Run sensitivity" if sensitivity_mode else "▶ Run council"
    run = st.sidebar.button(label, type="primary", use_container_width=True)

    st.sidebar.divider()
    up = st.sidebar.file_uploader("…or load a results/sensitivity JSON", type="json")
    if up is not None:
        try:
            st.session_state["result"] = json.load(up)
        except json.JSONDecodeError:
            st.sidebar.error("Not valid JSON.")

    if not run:
        return None
    if not topic.strip():
        st.sidebar.error("Enter a topic first.")
        return None
    cfg = RunConfig(
        topic=topic.strip(), tags=tags, require_all=require_all,
        ideas_to_evaluate=ideas, provider=provider, model=model or None,
        judges=judges, rounds=rounds, closing=closing,
        seed=(int(seed) if seed is not None else None),
        neutralize=neutralize, concurrency=concurrency, dedupe=dedupe,
        cache=cache, rag=rag, rag_k=rag_k, cards=cards, weights=weights,
    )
    return {"kind": "sensitivity" if sensitivity_mode else "council", "cfg": cfg,
            "paraphrases": paraphrases, "reframes": reframes}


def _run_streaming(worker, label: str) -> None:
    """Run ``worker(q, holder)`` in a background thread and drain its progress
    events into a live status panel. The worker only pushes events onto the
    queue and never touches Streamlit; only this (main) thread updates widgets.
    """
    q: queue.Queue = queue.Queue()
    holder: dict = {}

    def wrapped() -> None:
        try:
            worker(q, holder)
            q.put({"type": "done"})
        except Exception as err:  # noqa: BLE001 - report config/API errors in UI
            q.put({"type": "error", "msg": str(err)})

    threading.Thread(target=wrapped, daemon=True).start()

    with st.status(label, expanded=True) as status:
        bar = st.progress(0.0)
        line = st.empty()
        while True:
            ev = q.get()
            if ev["type"] == "progress":
                bar.progress(ev["frac"])
                line.markdown(f"*{ev['msg']}*")
                status.update(label=ev["msg"])
            elif ev["type"] == "done":
                bar.progress(1.0)
                status.update(label="Finished", state="complete", expanded=False)
                break
            else:  # error
                status.update(label="Run failed", state="error")
                st.error(ev["msg"])
                return
    st.session_state["result"] = holder.get("result")


def run_council(cfg: RunConfig) -> None:
    def worker(q, holder):
        holder["result"] = Council(cfg, progress_cb=q.put).run_session(write=False)
    _run_streaming(worker, "Convening the council…")


def run_sensitivity_ui(cfg: RunConfig, paraphrases: int, reframes: int) -> None:
    def worker(q, holder):
        from sensitivity import run_sensitivity
        holder["result"] = run_sensitivity(
            cfg, paraphrases, reframes, output="sensitivity.json", progress_cb=q.put)
    _run_streaming(worker, "Running wording-sensitivity analysis…")


def main() -> None:
    action = sidebar()
    if action is not None:
        if action["kind"] == "sensitivity":
            run_sensitivity_ui(action["cfg"], action["paraphrases"], action["reframes"])
        else:
            run_council(action["cfg"])

    res = st.session_state.get("result")
    if res and res.get("analysis") and res.get("runs"):
        render_sensitivity(res)
    elif res:
        render_result(res)
    else:
        st.title("🏛️ AI Council")
        st.markdown(
            "Generate, debate and evaluate ideas with a council of historical "
            "thinkers. **Configure a run in the sidebar and press ▶ Run council**, "
            "or upload a previous `results.json` / `sensitivity.json` to browse it."
        )
        st.info("Tip: filter the roster by tags (e.g. *economics*, *history*, "
                "*technology*) and re-weight the idea rubric to match what you care about.")


# Streamlit executes the script top-to-bottom on every interaction.
main()

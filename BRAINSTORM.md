# AI Council — Improvement Brainstorm & Roadmap

An analysis of the current codebase and a prioritised set of ideas for improving
the application, covering backend logic/prompts, model access, personas, data
output and UI. Items marked **[done]** were implemented in this change; the rest
are proposals.

---

## 1. Where the app stands today

The original `openaicouncil.py` is a single-file CLI that:

1. Loads economist "simulacra" from `CouncilMembers/*/` (each with
   `instructions.txt`, `info.txt` tags, and optional knowledge `Files/`).
2. Spins up an OpenAI **Assistant + thread** per member per side (for/against).
3. Has each member generate an idea, then argue for, against, and rebut.
4. A **Judge** assistant comments on each argument, and the comment is run
   through **VADER sentiment analysis** — positive text → +1, negative → −1.
5. Sums those ±1 signals to rank ideas and writes the winner to `best_option.txt`.

### Key problems identified

| # | Problem | Impact |
|---|---------|--------|
| 1 | Built on the **deprecated OpenAI Assistants API** (`beta.threads`, `beta.assistants`, `file_ids`). | The app is on a path to breaking entirely as OpenAI sunsets the beta. |
| 2 | **Sentiment ≠ quality.** VADER measures emotional tone of the judge's prose, not the logical merit of an argument. A judge saying "this is a *terrible*, *dangerous* fallacy" can score *positive* on words like "strong". | Scores are noisy and only loosely tied to argument quality. |
| 3 | **Single provider (OpenAI only)**, model hard-coded to `gpt-3.5-turbo` in `__main__`. | No access to cheaper/better models; locked to one vendor. |
| 4 | Uses a **module-global `council`** object inside methods (`council.client...`). | Fragile; breaks if more than one council exists; hidden coupling. |
| 5 | **Output is scattered** across `logs.txt`, `logs.csv`, `scoring.txt`, `sentiment.txt`, `ideas.txt`, `results.txt`, `best_option.txt`, all *appended* forever. | Hard to consume; results from different runs bleed together. |
| 6 | **Blocking `input()` prompts and CLI-only.** No arguments, no GUI. | Not scriptable; not shareable; poor UX. |
| 7 | **All 19 members are economists**; the framework is general but the roster isn't. | Limited range of perspectives. |
| 8 | Personas are short, hand-written one-liners of varying depth. | Members can sound generic and interchangeable. |
| 9 | Idea selection uses `random` without a seed; only 3 of N ideas evaluated. | Non-reproducible; most generated ideas are discarded unjudged. |
| 10 | No `requirements.txt`, no tests, secret read from an unusual env var name. | Hard to set up and trust. |

---

## 2. Implemented in this change

- **[done] OpenRouter support + provider abstraction** (`llm.py`). A single
  `LLMClient` speaks the OpenAI-compatible Chat Completions API, so the Council
  now runs against **OpenRouter** (hundreds of models, one key) *or* OpenAI.
  This also sidesteps the deprecated Assistants API entirely.
- **[done] Rubric-based judging** (`council.py`, `Judge`). The judge now returns
  structured JSON `{"score": 1-10, "reasoning": "..."}` on logic/evidence/rigour,
  replacing the VADER sentiment proxy. Robust parsing with a regex + neutral
  fallback so one malformed reply can't crash a run.
- **[done] 36 new members**, growing the roster 19 → 55 and broadening far
  beyond economics:
  - *Economics/innovation (8):* Schumpeter, Ostrom, Minsky, Galbraith, Sowell,
    Drucker, Christensen, Ha-Joon Chang.
  - *History (7):* Ibn Khaldun, Herodotus, Thucydides, Sima Qian, Gibbon,
    Braudel, Toynbee.
  - *Science (12):* Einstein, Newton, Darwin, Marie Curie, Galileo, Bohr,
    Feynman, Mendeleev, von Humboldt, Pasteur, Rosalind Franklin, Aristotle.
  - *Technology (9):* Tesla, Turing, Ada Lovelace, Leonardo da Vinci,
    von Neumann, Grace Hopper, Shannon, Wiener, Buckminster Fuller.

  The roster is deliberately multilingual/multicultural — figures who thought
  and wrote in Arabic, Greek, Chinese, French, German, Italian, Russian,
  Hungarian and Serbian are all included. New `master_tags.txt` tags cover
  history, science, technology, physics, chemistry, biology, mathematics,
  computing, engineering, medicine, ecology, geography and philosophy.
- **[done] Native-language "code-switching"** (`council.py`,
  `CODE_SWITCH_DIRECTIVE`). Non-English figures reason and argue in English (so
  scoring stays fair and transcripts stay readable) but weave in their authentic
  original-language key terms — e.g. Ibn Khaldun's *ʿaṣabiyyah* — glossing each
  in English on first use. The judge is instructed to score substance and
  neither reward nor penalise the code-switching. This was chosen over full
  native-language debate, which degrades reasoning quality for classical/low
  resource languages (Ancient Greek, Classical Chinese, medieval Arabic), risks
  language-biased scoring, and roughly doubles cost via a translate-back step.
  A future opt-in could enable true native-language generation + translation for
  *modern*-language figures only.
- **[done] Interview-driven persona builder** (`interview.py`). Generates probing
  questions, role-plays the figure's answers, and distils a rich
  `instructions.txt` + suggested tags — producing far more distinctive personas
  than a one-line prompt.
- **[done] Clean, scriptable CLI** with `--topic/--provider/--model/--tags/--ideas`
  and **structured JSON output** (`results.json`) containing every idea, the full
  debate transcript, per-argument scores and the winner.
- **[done] Multi-judge panel + median aggregation** (`council.py`, `JudgePanel`).
  A panel of judges with distinct temperaments (Sceptic, Pragmatist, Empiricist,
  Theorist, Generalist) each score every argument and idea; the aggregate is the
  **median**, robust to a single outlier judge. Size set via `--judges`
  (default 3); per-judge votes are stored in the transcript.
- **[done] Direct post-debate idea verdict, multi-axis.** Ideas are ranked by a
  judged verdict on the idea's *own merit* rather than by total argument volume
  (which previously rewarded the most *contentious* idea, not the best one). The
  verdict is a weighted **multi-axis rubric** — novelty, feasibility, evidence,
  logic, risk — median-aggregated per axis across the panel; users re-weight axes
  via `--weights`, and the per-axis vector is stored. Total argument quality is
  kept only as a tie-breaker.
- **[done] Reproducibility.** A `--seed` seeds the RNG and is forwarded to the API
  (`seed` param) for best-effort determinism, and is recorded in `results.json`
  alongside model, temperature, members and judges.
- **[done] Topic neutralisation + wording-sensitivity analysis** (`framing.py`,
  `sensitivity.py`). Every run first debiases the topic into a neutral, open
  question (loaded terms/presuppositions detected and recorded; neutrality
  verified; `--no-neutralize` opts out). `sensitivity.py` runs the council across
  a neutral baseline plus meaning-preserving **paraphrases** and deliberately
  re-slanted **reframes** (seed fixed), separating **lexical robustness** (answer
  should be stable under paraphrase — instability is noise) from **framing
  sensitivity** (answer moving under reframing is a genuine finding). Idea
  overlap uses embeddings with a lexical fallback.
- **[done] Concurrency, de-duplication, caching and cost accounting** (Tier 2).
  Independent debate/scoring calls fan out across threads (`--concurrency`);
  generated ideas are de-duplicated before the debate via embeddings with a
  lexical fallback (`similarity.py`, `--no-dedupe`); identical requests are
  served from a thread-safe on-disk cache (`--no-cache`); and per-run token
  usage + estimated USD cost are accumulated and recorded in `results.json`.
- **[done] `requirements.txt`** and expanded `master_tags.txt`.

---

## 3. Backend logic & prompt improvements (proposals)

- **[done] Multi-judge panels + score aggregation.** A `JudgePanel` of judges with
  distinct temperaments (Sceptic, Pragmatist, Empiricist, Theorist, Generalist)
  median-aggregates each rubric score; `--judges` sets the size, per-judge votes
  are stored.
- **[done] Multi-dimensional rubric.** The idea verdict is scored on five axes
  (novelty, feasibility, evidence, logic, risk), median-aggregated per axis
  across the panel and combined into a weighted overall verdict; users re-weight
  axes with `--weights` (e.g. `novelty=2,feasibility=1.5`), and the per-axis
  vector is stored in `results.json`. (Argument scoring stays a single quality
  number — it is only the debate-substance tie-breaker.)
- **[done] Score the *idea*, not just the arguments.** Ideas are ranked by a direct
  post-debate verdict on the idea's own merit (informed by the strongest points
  each side made); total argument quality is now only a tie-breaker.
- **Structured debate rounds.** Opening → cross-examination → closing, with a
  configurable number of rounds, instead of a single arg+rebut exchange.
- **[done] De-duplication / clustering of ideas.** Generated ideas are embedded
  and greedily clustered (`similarity.py`) to drop near-duplicates before the
  expensive debate phase; `--dedupe-threshold` tunes it, `--no-dedupe` disables.
- **[done] Retrieval / RAG for knowledge files** (`rag.py`). Each member's `Files/`
  corpus (`.txt`/`.md`/`.pdf` via pdfminer.six) is chunked and embedded (cached),
  and the passages most relevant to the question are retrieved and injected into
  the member's prompt so they argue from their own corpus (`--rag-k`, `--no-rag`).
  The new history/science/technology members and newer economists were given
  curated `key_ideas.md` reference digests (their primary texts are under
  copyright / unavailable offline); see `CouncilMembers/CORPUS.md`.
- **[done] Cost & token accounting.** Per-run tokens and an estimated USD cost are
  accumulated from the API usage field and recorded/printed; pricing is
  extensible via `LLM_PRICING_JSON`.
- **[done] Reproducibility.** `--seed` seeds the RNG, is forwarded to the API for
  best-effort determinism, and is recorded with model/temperature/members/judges
  in `results.json`.
- **[done] Async / concurrent calls.** Independent debate/scoring calls fan out
  across a thread pool (`pmap`, `--concurrency`), cutting wall-clock time.
- **[done] Caching.** Identical `(model, messages, temperature, response_format,
  seed)` requests are served from a thread-safe on-disk cache (`--no-cache`).

## 4. Persona & "interview" improvements

- **Interview depth knobs.** Follow-up questions that dig into the *weakest* prior
  answer; adversarial questions that force the persona to defend against critics.
- **Self-consistency check.** After building a persona, run a few sanity
  questions and have a judge rate how well answers match the known figure.
- **Persona knowledge cards.** Alongside `instructions.txt`, generate a short list
  of the figure's signature concepts, canonical quotes and rivals, injected as
  few-shot flavour.
- **[done] Broaden the roster** beyond economics: historians, scientists and
  technologists are now included, so the Council can tackle non-economic topics.
  Further breadth (ethicists, artists, legal thinkers, non-Western political
  philosophers) would extend this further.

## 5. Data output & evaluation

- **Single JSON per run** (done) + optional Markdown report renderer
  (`results.json` → a readable debate write-up).
- **Run manifest / history folder.** Write each run to `runs/<timestamp>/` instead
  of appending to shared files forever.
- **Leaderboard across runs.** Track which members' ideas win most often and which
  judges are harshest/most lenient.

## 6. UI improvements

- **Web UI (recommended next step).** A small **Streamlit** or **FastAPI + React**
  front end: enter a topic, pick provider/model and tags, watch ideas and the
  debate stream in live, and browse the scored transcript. `results.json` is
  already shaped to drive this directly.
- **Streaming output** so the user sees arguments as they generate rather than
  waiting for the whole run.
- **Interactive member/tag picker** (checkbox list from `master_tags.txt`) instead
  of numeric console prompts.
- **Shareable result pages** and export to PDF/Markdown.
- **"Add a member" flow** that calls `interview.py` from the UI and previews the
  generated persona before saving.

## 7. Engineering hygiene

- Add `pytest` unit tests for persona loading, tag filtering and judge parsing
  (pure functions, no network needed — see the smoke tests used during this
  change).
- Add a `.env.example`, type hints throughout (mostly done in the new modules),
  and a `pyproject.toml`.
- CI to lint and run the offline tests.

---

## Suggested next milestone

The result-quality bundle (multi-judge panel, direct idea verdict, seed), the
full Tier 2 practicality bundle (concurrency, de-dup, caching, cost accounting),
RAG over member corpora, and the weighted multi-axis rubric are **done**.
Remaining high-value work, in order:

1. Streamlit/web UI over `results.json` (biggest UX win).
2. Structured multi-round debate + persona knowledge cards.
3. Add real primary texts for the curated members (drop into their `Files/`).

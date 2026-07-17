# Member corpora (RAG source material)

Each member folder may contain a `Files/` directory. At runtime `rag.py` reads
these files, chunks and embeds them, and retrieves the most relevant passages to
ground that member's arguments in their own material (see the RAG section of the
main README). Supported file types: `.txt`, `.md`, and `.pdf`.

There are **two kinds of source** in this repo:

1. **Primary texts** — real books/papers/letters by (or foundational to) the
   member. These are the original corpora that shipped with the project, e.g.:
   - Adam Smith — *The Wealth of Nations*
   - Karl Marx — *Capital* Vol. I, *The Communist Manifesto*
   - J. M. Keynes — *The General Theory*
   - David Ricardo — *Principles of Political Economy and Taxation*, letters
   - Thomas Malthus — *Principle of Population* and others
   - Milton Friedman — *Capitalism and Freedom*
   - Warren Buffett — Berkshire chairman's letters + *The Intelligent Investor*
   - Hayek, Mises, Veblen, Commons, Marshall, and the Judge's argument guides.

2. **Curated reference notes** (`key_ideas.md`) — for the members added later
   (the history / science / technology figures and the newer economists), the
   primary texts could not be included here: many are still under copyright, and
   the build environment has no access to public-domain text archives. Each of
   these members therefore has a **curated digest** of their major works and core
   ideas, clearly headed as reference notes and containing **no fabricated
   quotations**. They give RAG real, accurate material to ground on without
   misattributing verbatim text.

## Adding real primary texts

To upgrade any curated member to primary sources, drop the real `.txt`/`.pdf`
files into that member's `Files/` folder (public-domain works from e.g. Project
Gutenberg or the Internet Archive are ideal, licensing permitting). RAG will
index them automatically on the next run; you can leave or remove the
`key_ideas.md` digest. Indexing is bounded (see `rag.py`: `MAX_PDF_PAGES`,
`MAX_CHARS_PER_FILE`, `MAX_CHUNKS_PER_MEMBER`) so large books stay affordable,
and embeddings are cached so re-runs don't re-pay.

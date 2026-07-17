"""Vendor-neutral retrieval over each council member's ``Files/`` corpus.

The original app fed each member's documents (Buffett's shareholder letters,
Keynes' *General Theory*, ...) to the model via OpenAI's Assistants retrieval,
which we dropped when moving to Chat Completions. This restores that grounding
provider-agnostically: read a member's files, chunk them, embed the chunks
(cached on disk via ``LLMClient``), and at debate time retrieve the passages
most relevant to the question and inject them into the prompt so the member
argues from their own corpus.

Text is read from .txt/.md directly and from .pdf via pdfminer.six. Everything
degrades gracefully: if a member has no files, or embeddings/PDF-extraction are
unavailable, retrieval is simply skipped for that member.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from similarity import _cosine

# Bounds to keep indexing affordable on very large corpora (e.g. multi-MB books).
MAX_PDF_PAGES = 60
MAX_CHARS_PER_FILE = 300_000
MAX_CHUNKS_PER_MEMBER = 200
CHUNK_WORDS = 180
CHUNK_OVERLAP = 30
_EMBED_BATCH = 64

TEXT_EXTS = {".txt", ".md"}


@dataclass
class Chunk:
    source: str
    text: str
    embedding: list[float] | None = None


def _read_pdf(path: str) -> str:
    try:
        from pdfminer.high_level import extract_text

        return extract_text(path, maxpages=MAX_PDF_PAGES) or ""
    except Exception as err:  # noqa: BLE001 - pdf extraction is best-effort
        print(f"  [rag] could not read {os.path.basename(path)}: {err}")
        return ""


def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTS:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError as err:
            print(f"  [rag] could not read {os.path.basename(path)}: {err}")
            return ""
    if ext == ".pdf":
        return _read_pdf(path)
    return ""  # skip unknown types


def chunk_text(text: str, source: str) -> list[Chunk]:
    """Split into ~CHUNK_WORDS-word overlapping chunks on whitespace."""
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if not words or words == [""]:
        return []
    chunks: list[Chunk] = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + CHUNK_WORDS]).strip()
        if len(piece) > 40:  # drop trivially short tails
            chunks.append(Chunk(source=source, text=piece))
    return chunks


class CorpusIndex:
    """An embedded corpus for one member, supporting top-k retrieval."""

    def __init__(self, name: str, chunks: list[Chunk], client):
        self.name = name
        self.chunks = chunks
        self.client = client

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        if not self.chunks or k <= 0:
            return []
        try:
            q = self.client.embed([query])[0]
        except Exception:  # noqa: BLE001 - query embedding best-effort
            return []
        ranked = sorted(
            self.chunks,
            key=lambda c: _cosine(q, c.embedding) if c.embedding else -1.0,
            reverse=True,
        )
        return ranked[:k]


def _embed_chunks(chunks: list[Chunk], client) -> bool:
    """Embed chunks in batches (cached by LLMClient). Returns False on failure."""
    for i in range(0, len(chunks), _EMBED_BATCH):
        batch = chunks[i : i + _EMBED_BATCH]
        try:
            vectors = client.embed([c.text for c in batch])
        except Exception as err:  # noqa: BLE001 - embeddings are best-effort
            print(f"  [rag] embeddings unavailable ({err}); skipping RAG for this member")
            return False
        for c, v in zip(batch, vectors):
            c.embedding = v
    return True


def build_index(member_dir: str, client, name: str | None = None) -> CorpusIndex | None:
    """Build a retrieval index for a member, or None if there is nothing to index."""
    files_dir = os.path.join(member_dir, "Files")
    if not os.path.isdir(files_dir):
        return None
    name = name or os.path.basename(member_dir.rstrip("/"))

    chunks: list[Chunk] = []
    for fname in sorted(os.listdir(files_dir)):
        path = os.path.join(files_dir, fname)
        if not os.path.isfile(path):
            continue
        text = _read_file(path)[:MAX_CHARS_PER_FILE]
        if not text.strip():
            continue
        chunks.extend(chunk_text(text, source=fname))
        if len(chunks) >= MAX_CHUNKS_PER_MEMBER:
            chunks = chunks[:MAX_CHUNKS_PER_MEMBER]
            break

    if not chunks:
        return None
    if not _embed_chunks(chunks, client):
        return None
    return CorpusIndex(name, chunks, client)

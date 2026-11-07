"""Text-similarity helpers shared by idea de-duplication and sensitivity analysis.

Uses embeddings when the provider/model supports them, and falls back to a
lexical similarity otherwise so the features work anywhere.
"""

from __future__ import annotations

import difflib
import math
from typing import Callable


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def similarity_fn(texts: list[str], client) -> tuple[Callable[[int, int], float], str]:
    """Return ``(sim(i, j) -> float in [0, 1], method_name)`` for ``texts``.

    Tries embeddings once; on any failure falls back to a lexical ratio. The
    returned closure is pure/read-only, so it is safe to call from threads.
    """
    try:
        embs = client.embed(texts)

        def sim(i: int, j: int) -> float:
            return _cosine(embs[i], embs[j])

        return sim, "embeddings"
    except Exception as err:  # noqa: BLE001 - embeddings are best-effort
        print(f"[similarity] embeddings unavailable ({err}); using lexical similarity")

        def sim(i: int, j: int) -> float:
            return difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()

        return sim, "lexical"


def dedupe(
    texts: list[str], client, threshold: float = 0.85
) -> tuple[list[int], dict[int, list[int]], str]:
    """Greedily cluster near-duplicate texts.

    Returns ``(kept_indices, clusters, method)`` where ``clusters`` maps each
    kept (representative) index to the list of indices merged into it. Order is
    preserved, so the first occurrence of each distinct idea is the representative.
    """
    n = len(texts)
    if n <= 1:
        return list(range(n)), {i: [] for i in range(n)}, "none"

    sim, method = similarity_fn(texts, client)
    kept: list[int] = []
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        rep = next((k for k in kept if sim(i, k) >= threshold), None)
        if rep is None:
            kept.append(i)
            clusters[i] = []
        else:
            clusters[rep].append(i)
    return kept, clusters, method

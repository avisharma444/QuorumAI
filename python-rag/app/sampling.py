"""Beginning-Middle-End (BME) document sampling, as in the reference ingest router.

The sample is what the Go ingestion agent summarizes. It works on the ordered
list of chunk texts, so it stays compatible with the layout-aware chunker
planned for a later phase (which also produces ordered chunks with content).
"""
from __future__ import annotations

from typing import Sequence

PREVIEW_CHAR_LIMIT = 15_000
_EDGE = 3  # chunks taken from the beginning, middle and end


def bme_indices(n_chunks: int) -> list[int]:
    """Chunk indices in document order.

    N <= 9: every chunk. N > 9: chunks[:3] + chunks[N//2-1 : N//2+2] + chunks[-3:].
    For N >= 10 the three windows never overlap.
    """
    if n_chunks < 0:
        raise ValueError("n_chunks must be non-negative")
    if n_chunks <= 3 * _EDGE:
        return list(range(n_chunks))
    mid = n_chunks // 2
    return list(range(_EDGE)) + list(range(mid - 1, mid + 2)) + list(range(n_chunks - _EDGE, n_chunks))


def bme_preview(chunks: Sequence[str], limit: int = PREVIEW_CHAR_LIMIT) -> str:
    """Join the BME sample with newlines and cap it at `limit` characters."""
    preview = "\n".join(chunks[i] for i in bme_indices(len(chunks)))
    return preview[:limit]

"""Voyage rerank-2.5 — second-stage precision over the RRF candidate pool.

Bi-encoders (our embeddings) score query and document independently and
compare cosines. A cross-encoder runs a transformer over `(query, doc)` as a
single concatenated input — much better at catching the "weapon ESC vs drive
ESC" kind of mismatch our embeddings would otherwise flatten. Cost is O(N)
per query (no precomputation possible), which is why it sits AFTER cheap
recall, ranking ~100 survivors instead of the full 7.7K corpus.

We use Voyage's hosted `rerank-2.5` — same vendor as the embeddings, so a
single VOYAGE_API_KEY covers both. Cost is ~$0.05/M tokens; for our 100-doc
pool with ~600 tokens each that's ~$0.003 per query.
"""
from __future__ import annotations

import os
import time

import voyageai

RERANK_MODEL = "rerank-2.5"


def _client() -> voyageai.Client:
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set. Put it in .env or export it in your shell."
        )
    return voyageai.Client(api_key=api_key)


def rerank(
    query: str,
    docs: list[str],
    *,
    top_n: int | None = None,
    client: voyageai.Client | None = None,
) -> list[tuple[int, float]]:
    """Score `docs` against `query`. Returns `[(orig_index, score)]` sorted desc.

    Indices are positions in the input list, so callers can map back to their
    own metadata (chunk_id, etc.).
    """
    if not docs:
        return []
    client = client or _client()

    delay = 1.0
    for attempt in range(5):
        try:
            resp = client.rerank(
                query=query,
                documents=docs,
                model=RERANK_MODEL,
                top_k=top_n or len(docs),
            )
            return [(r.index, r.relevance_score) for r in resp.results]
        except Exception as e:
            msg = str(e).lower()
            transient = any(
                k in msg
                for k in ("rate", "timeout", "temporarily", "502", "503", "504")
            )
            if attempt == 4 or not transient:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")

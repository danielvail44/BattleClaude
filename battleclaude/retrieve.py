"""Hybrid retrieval with optional cross-encoder reranking.

Two stages:
  1. RECALL — vector similarity + BM25, fused via Reciprocal Rank Fusion.
     Cheap, runs over the whole corpus, cast a wide net.
  2. PRECISION — Voyage `rerank-2.5` cross-encoder over the RRF top-N.
     Expensive but accurate; replaces the simple `fused_score` ordering.

RRF (chosen over learned-weight fusion) needs no tuning and is robust across
very different score distributions. Reranking (chosen over leaving fusion as
the final ranker) catches semantic mismatches the bi-encoder embedding can't
see — "weapon ESC vs drive ESC" being the canonical example here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np

from .bm25 import DEFAULT_BM25_PATH, load_bm25, tokenize
from .embed import embed_query
from .rerank import rerank as voyage_rerank

RRF_K = 60                # RRF denominator constant (60 is standard)
VECTOR_POOL = 200         # top-K retrieved by vectors before fusion
BM25_POOL = 200           # top-K retrieved by BM25 before fusion
RERANK_POOL = 100         # how many fused candidates to feed the reranker


@dataclass
class Hit:
    chunk_id: str
    channel_name: str
    start_ts: object
    end_ts: object
    message_count: int
    text: str
    vec_rank: int | None
    bm25_rank: int | None
    fused_score: float
    rerank_score: float | None = None


def _load_all_embeddings(conn: duckdb.DuckDBPyConnection) -> tuple[list[str], np.ndarray]:
    """Return (chunk_ids, matrix) where matrix is L2-normalized for cosine = dot."""
    rows = conn.execute(
        "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY chunk_id"
    ).fetchall()
    if not rows:
        raise RuntimeError("No chunk embeddings found. Run `battleclaude.embed` first.")
    chunk_ids = [r[0] for r in rows]
    mat = np.asarray([r[1] for r in rows], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return chunk_ids, mat


def _vector_rank(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    pool: int,
) -> list[tuple[str, int]]:
    chunk_ids, mat = _load_all_embeddings(conn)
    q_vec = np.asarray(embed_query(query), dtype=np.float32)
    q_vec /= max(float(np.linalg.norm(q_vec)), 1e-12)
    sims = mat @ q_vec
    k = min(pool, len(chunk_ids))
    # top-k via argpartition then sort the slice
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(chunk_ids[i], rank) for rank, i in enumerate(top_idx, start=1)]


def _bm25_rank(
    query: str,
    pool: int,
    bm25_path: Path,
) -> list[tuple[str, int]]:
    chunk_ids, bm25 = load_bm25(bm25_path)
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    k = min(pool, len(chunk_ids))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(chunk_ids[i], rank) for rank, i in enumerate(top_idx, start=1)]


def _fuse(
    vector_ranking: list[tuple[str, int]],
    bm25_ranking: list[tuple[str, int]],
    k: int = RRF_K,
) -> list[tuple[str, float, int | None, int | None]]:
    vec_rank: dict[str, int] = {cid: r for cid, r in vector_ranking}
    bm25_rank: dict[str, int] = {cid: r for cid, r in bm25_ranking}
    all_ids = set(vec_rank) | set(bm25_rank)
    scored = []
    for cid in all_ids:
        vr = vec_rank.get(cid)
        br = bm25_rank.get(cid)
        score = 0.0
        if vr is not None:
            score += 1.0 / (k + vr)
        if br is not None:
            score += 1.0 / (k + br)
        scored.append((cid, score, vr, br))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def retrieve(
    conn: duckdb.DuckDBPyConnection,
    query: str,
    top_k: int = 30,
    *,
    bm25_path: Path = DEFAULT_BM25_PATH,
    use_rerank: bool = True,
    rerank_pool: int = RERANK_POOL,
) -> list[Hit]:
    vec_ranking = _vector_rank(conn, query, VECTOR_POOL)
    bm25_ranking = _bm25_rank(query, BM25_POOL, bm25_path)
    fused_all = _fuse(vec_ranking, bm25_ranking)

    if not fused_all:
        return []

    # Reranking takes the top RERANK_POOL fused candidates and re-orders them
    # with a cross-encoder. Without reranking, we just truncate at top_k.
    pool_size = max(top_k, rerank_pool) if use_rerank else top_k
    fused = fused_all[:pool_size]

    ids = [cid for cid, *_ in fused]
    placeholders = ", ".join(["?"] * len(ids))
    rows = conn.execute(
        f"""
        SELECT chunk_id, channel_name, start_timestamp, end_timestamp,
               message_count, text
        FROM chunks
        WHERE chunk_id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    by_id = {r[0]: r for r in rows}

    # Build candidate Hits in fused order; we may re-sort below.
    candidates: list[Hit] = []
    for cid, score, vr, br in fused:
        r = by_id.get(cid)
        if r is None:
            continue
        candidates.append(
            Hit(
                chunk_id=r[0],
                channel_name=r[1],
                start_ts=r[2],
                end_ts=r[3],
                message_count=r[4],
                text=r[5],
                vec_rank=vr,
                bm25_rank=br,
                fused_score=score,
            )
        )

    if not use_rerank or not candidates:
        return candidates[:top_k]

    # Stage 2: cross-encoder rerank over the candidate pool.
    rerank_results = voyage_rerank(
        query=query,
        docs=[h.text for h in candidates],
        top_n=top_k,
    )
    reranked: list[Hit] = []
    for orig_idx, score in rerank_results:
        h = candidates[orig_idx]
        h.rerank_score = score
        reranked.append(h)
    return reranked

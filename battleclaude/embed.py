"""Embedding client for voyage-context-3 (contextualized chunk embeddings).

voyage-context-3 embeds each chunk conditioned on the other chunks in the same
"document" — ideal for short Discord messages whose meaning depends on the
surrounding conversation. We treat each `session_id` as a document, which
keeps multi-hour design discussions together while splitting unrelated later
chat.

Per-request API limits (from Voyage docs): 120K total tokens, 16K total
chunks, 1,000 inputs (documents). The batcher below packs as many whole
sessions into each request as those limits allow.

Resumability: we only re-embed chunks where `embedding IS NULL`, so a failed
run can be rerun without re-paying for completed sessions.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb
import voyageai
from tqdm import tqdm

VOYAGE_MODEL = "voyage-context-3"
VOYAGE_DIM = 1024

# Voyage contextualized_embed per-request caps (hard limits: 120K tokens, 16K
# chunks, 1000 inputs). We keep headroom because we size with Voyage's real
# tokenizer — no need to over-reserve.
MAX_TOKENS_PER_REQUEST = 110_000
MAX_CHUNKS_PER_REQUEST = 15_000
MAX_DOCUMENTS_PER_REQUEST = 900

# Per-document (per-session) caps. If a session exceeds these, we split it
# into contiguous sub-documents before batching. Voyage's hard per-document
# cap is 32K tokens; 30K leaves room for the model's prompt prefix.
MAX_TOKENS_PER_DOCUMENT = 30_000
MAX_CHUNKS_PER_DOCUMENT = 200


def _client() -> voyageai.Client:
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY is not set. Put it in .env or export it in your shell."
        )
    return voyageai.Client(api_key=api_key)


def _call_with_retry(
    client: voyageai.Client,
    inputs: list[list[str]],
    input_type: str,
) -> list[list[list[float]]]:
    """Call contextualized_embed with exponential backoff on transient failures.

    Returns a list-per-document of list-per-chunk embeddings.
    """
    delay = 1.0
    for attempt in range(5):
        try:
            resp = client.contextualized_embed(
                inputs=inputs,
                model=VOYAGE_MODEL,
                input_type=input_type,
                output_dimension=VOYAGE_DIM,
            )
            return [r.embeddings for r in resp.results]
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


def _split_oversized_session(
    chunk_ids: list[str],
    texts: list[str],
    token_counts: list[int],
) -> list[tuple[list[str], list[str], list[int]]]:
    """Split a single session into sub-documents that each fit per-doc caps.

    Returns a list of (chunk_ids, texts, token_counts) triples. Short sessions
    pass through as a single triple.
    """
    total_tokens = sum(token_counts)
    if total_tokens <= MAX_TOKENS_PER_DOCUMENT and len(texts) <= MAX_CHUNKS_PER_DOCUMENT:
        return [(chunk_ids, texts, token_counts)]

    subdocs: list[tuple[list[str], list[str], list[int]]] = []
    cur_ids: list[str] = []
    cur_texts: list[str] = []
    cur_tokens: list[int] = []
    cur_sum = 0
    for cid, text, t in zip(chunk_ids, texts, token_counts):
        # A single chunk larger than the doc cap can't be split further here —
        # just let it through as its own doc; the API will reject if truly over.
        if cur_ids and (
            cur_sum + t > MAX_TOKENS_PER_DOCUMENT
            or len(cur_ids) >= MAX_CHUNKS_PER_DOCUMENT
        ):
            subdocs.append((cur_ids, cur_texts, cur_tokens))
            cur_ids, cur_texts, cur_tokens, cur_sum = [], [], [], 0
        cur_ids.append(cid)
        cur_texts.append(text)
        cur_tokens.append(t)
        cur_sum += t
    if cur_ids:
        subdocs.append((cur_ids, cur_texts, cur_tokens))
    return subdocs


def _pack_requests(
    documents: list[tuple[list[str], list[str], list[int]]],
) -> list[list[tuple[list[str], list[str], list[int]]]]:
    """Pack sub-documents into requests that fit the per-request caps."""
    requests: list[list[tuple[list[str], list[str], list[int]]]] = []
    cur: list[tuple[list[str], list[str], list[int]]] = []
    cur_tokens = 0
    cur_chunks = 0
    for doc in documents:
        doc_tokens = sum(doc[2])
        doc_chunks = len(doc[1])
        too_big = (
            cur
            and (
                cur_tokens + doc_tokens > MAX_TOKENS_PER_REQUEST
                or cur_chunks + doc_chunks > MAX_CHUNKS_PER_REQUEST
                or len(cur) >= MAX_DOCUMENTS_PER_REQUEST
            )
        )
        if too_big:
            requests.append(cur)
            cur, cur_tokens, cur_chunks = [], 0, 0
        cur.append(doc)
        cur_tokens += doc_tokens
        cur_chunks += doc_chunks
    if cur:
        requests.append(cur)
    return requests


def embed_all_chunks(
    conn: duckdb.DuckDBPyConnection,
    *,
    only_missing: bool = True,
) -> int:
    """Embed every chunk whose embedding is NULL, grouping by session_id."""
    where = "WHERE embedding IS NULL" if only_missing else ""
    rows = conn.execute(
        f"""
        SELECT chunk_id, session_id, text
          FROM chunks
          {where}
         ORDER BY session_id, start_timestamp, chunk_id
        """
    ).fetchall()
    if not rows:
        print("  no chunks to embed")
        return 0

    client = _client()

    # Ask Voyage's tokenizer for the token count of every chunk up front.
    # `count_tokens` returns the sum for the whole batch, so we call per-chunk
    # to get per-chunk counts. First call downloads the tokenizer (~100 MB)
    # into the HuggingFace cache; subsequent runs hit the cache.
    print(f"  counting tokens for {len(rows):,} chunks with Voyage tokenizer...")
    token_counts: list[int] = []
    pbar = tqdm(total=len(rows), unit=" chunk", unit_scale=True, leave=False)
    for _, _, text in rows:
        token_counts.append(client.count_tokens([text], model=VOYAGE_MODEL))
        pbar.update(1)
    pbar.close()

    # Group by session_id, preserving order.
    sessions: dict[str, tuple[list[str], list[str], list[int]]] = {}
    session_order: list[str] = []
    for (chunk_id, session_id, text), tok in zip(rows, token_counts):
        if session_id not in sessions:
            sessions[session_id] = ([], [], [])
            session_order.append(session_id)
        sessions[session_id][0].append(chunk_id)
        sessions[session_id][1].append(text)
        sessions[session_id][2].append(tok)

    # Split any oversized sessions and flatten to a list of sub-documents.
    documents: list[tuple[list[str], list[str], list[int]]] = []
    for sid in session_order:
        ids, texts, toks = sessions[sid]
        documents.extend(_split_oversized_session(ids, texts, toks))

    requests = _pack_requests(documents)

    total_chunks = sum(len(d[1]) for d in documents)
    total_tokens = sum(sum(d[2]) for d in documents)
    print(
        f"  embedding {total_chunks:,} chunks ({total_tokens:,} tokens) across "
        f"{len(documents):,} documents ({len(session_order):,} sessions) in "
        f"{len(requests)} API request(s)"
    )

    pbar = tqdm(total=total_chunks, unit=" chunk", unit_scale=True)
    try:
        for req in requests:
            inputs = [texts for _ids, texts, _toks in req]
            embeddings_per_doc = _call_with_retry(client, inputs, input_type="document")
            if len(embeddings_per_doc) != len(req):
                raise RuntimeError(
                    f"Voyage returned {len(embeddings_per_doc)} docs for {len(req)} inputs"
                )
            for (ids, _texts, _toks), doc_embeddings in zip(req, embeddings_per_doc):
                if len(doc_embeddings) != len(ids):
                    raise RuntimeError(
                        f"Voyage returned {len(doc_embeddings)} vectors for a {len(ids)}-chunk doc"
                    )
                for cid, vec in zip(ids, doc_embeddings):
                    conn.execute(
                        "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
                        [vec, cid],
                    )
                pbar.update(len(ids))
    finally:
        pbar.close()

    return total_chunks


def embed_query(query: str, client: voyageai.Client | None = None) -> list[float]:
    """Embed a single query with voyage-context-3."""
    client = client or _client()
    resp = client.contextualized_embed(
        inputs=[[query]],
        model=VOYAGE_MODEL,
        input_type="query",
        output_dimension=VOYAGE_DIM,
    )
    return resp.results[0].embeddings[0]


def main() -> None:
    from argparse import ArgumentParser
    from dotenv import load_dotenv

    from .db import DEFAULT_DB_PATH, connect

    load_dotenv()

    parser = ArgumentParser(description="Embed chunks with Voyage voyage-context-3.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--all",
        action="store_true",
        help="re-embed even chunks that already have embeddings",
    )
    args = parser.parse_args()

    conn = connect(args.db)
    n = embed_all_chunks(conn, only_missing=not args.all)
    print(f"Embedded {n:,} chunks")
    conn.close()


if __name__ == "__main__":
    main()

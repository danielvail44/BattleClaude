"""Build and persist a BM25 index over chunk text.

BM25 pairs with semantic embeddings in retrieval: embeddings catch paraphrase
and synonymy, BM25 is the safety net for rare jargon (VESC, FOC, Wraith 32,
kg·m²) that gets flattened by embeddings.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import duckdb
from rank_bm25 import BM25Okapi

DEFAULT_BM25_PATH = Path("battleclaude.bm25.pkl")

# Tokenizer choices matter for combat-robotics jargon:
#  - keep dots (VESC 6.0, M4.2), plus signs (V+), hyphens (Wraith-32)
#  - split on whitespace and punctuation otherwise
#  - lowercase for case-insensitive match
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._+\-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_bm25(conn: duckdb.DuckDBPyConnection, out_path: Path = DEFAULT_BM25_PATH) -> tuple[int, Path]:
    """Build a BM25 index from chunks.text and pickle it to disk."""
    rows = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    if not rows:
        raise RuntimeError("No chunks to index. Run `battleclaude.chunk` first.")

    chunk_ids = [r[0] for r in rows]
    tokenized = [tokenize(r[1]) for r in rows]

    bm25 = BM25Okapi(tokenized)

    with open(out_path, "wb") as f:
        pickle.dump({"chunk_ids": chunk_ids, "bm25": bm25}, f, protocol=pickle.HIGHEST_PROTOCOL)

    return len(chunk_ids), out_path


def load_bm25(path: Path = DEFAULT_BM25_PATH) -> tuple[list[str], BM25Okapi]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["chunk_ids"], data["bm25"]


def main() -> None:
    from argparse import ArgumentParser

    from .db import DEFAULT_DB_PATH, connect

    parser = ArgumentParser(description="Build and pickle a BM25 index over chunk text.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_BM25_PATH)
    args = parser.parse_args()

    conn = connect(args.db)
    n, path = build_bm25(conn, args.out)
    conn.close()
    print(f"Built BM25 index over {n:,} chunks -> {path}")


if __name__ == "__main__":
    main()

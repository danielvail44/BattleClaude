"""One-shot indexing pipeline: chunk -> embed -> bm25.

Run after `battleclaude.ingest`. This is the slowest step because embedding
calls Voyage over the network; subsequent runs only re-embed chunks whose
`embedding` column is still NULL.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

from .bm25 import DEFAULT_BM25_PATH, build_bm25
from .chunk import build_chunks
from .db import DEFAULT_DB_PATH, connect
from .embed import embed_all_chunks


def main() -> None:
    load_dotenv()

    parser = ArgumentParser(description="Build chunks, embeddings, and BM25 index.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="rebuild chunks from messages (drops existing chunks and embeddings)",
    )
    parser.add_argument(
        "--reembed-all",
        action="store_true",
        help="re-embed every chunk even if it already has an embedding",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="skip the Voyage embedding step (useful for offline BM25-only iterations)",
    )
    args = parser.parse_args()

    conn = connect(args.db)

    # 1. Chunks
    existing = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if args.rechunk or existing == 0:
        print("== Chunking messages ==")
        build_chunks(conn)
    else:
        print(f"== Chunks: {existing:,} already built (use --rechunk to rebuild) ==")

    # 2. Embeddings
    if args.skip_embed:
        print("== Skipping embeddings (--skip-embed) ==")
    else:
        print("== Embedding chunks ==")
        embed_all_chunks(conn, only_missing=not args.reembed_all)

    # 3. BM25
    print("== Building BM25 index ==")
    n, path = build_bm25(conn, args.bm25)
    print(f"BM25 index: {n:,} chunks -> {path}")

    conn.close()
    print("\nIndex build complete. You can now run `battleclaude-ask \"<question>\"`.")


if __name__ == "__main__":
    main()

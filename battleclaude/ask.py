"""CLI: retrieve + synthesize an answer for a natural-language question.

    python -m battleclaude.ask "What do 12lb horizontals range in for weapon MOI?"
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv


def _force_utf8_stdout() -> None:
    """Discord content contains emoji; Windows' default cp1252 console crashes on them."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

from .bm25 import DEFAULT_BM25_PATH
from .db import DEFAULT_DB_PATH, connect
from .retrieve import retrieve
from .synthesize import synthesize

NHRL_GUILD_ID = "651601084019900483"


def _discord_jump_url(channel_id: str, message_id: str) -> str:
    return f"https://discord.com/channels/{NHRL_GUILD_ID}/{channel_id}/{message_id}"


def main() -> None:
    _force_utf8_stdout()
    load_dotenv()

    parser = ArgumentParser(description="Ask BattleClaude a question.")
    parser.add_argument("question", type=str, help="the question to ask (wrap in quotes)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument("--top-k", type=int, default=30, help="chunks sent to Claude")
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="print the retrieved chunk headers and ranks before the answer",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="skip Voyage rerank (use raw RRF fusion order — useful for A/B comparison)",
    )
    args = parser.parse_args()

    conn = connect(args.db)

    stage = "RRF only" if args.no_rerank else "RRF + Voyage rerank"
    print(f"Retrieving top {args.top_k} chunks ({stage}) for: {args.question!r}\n")
    hits = retrieve(
        conn,
        args.question,
        top_k=args.top_k,
        bm25_path=args.bm25,
        use_rerank=not args.no_rerank,
    )

    if args.show_chunks:
        print("== Retrieved ==")
        for i, h in enumerate(hits, start=1):
            start = h.start_ts.strftime("%Y-%m-%d %H:%M") if h.start_ts else "?"
            rerank_str = f"  rerank={h.rerank_score:.3f}" if h.rerank_score is not None else ""
            print(
                f"{i:>2}. [chunk:{h.chunk_id}] #{h.channel_name}  {start}  "
                f"vec={h.vec_rank}  bm25={h.bm25_rank}  rrf={h.fused_score:.4f}{rerank_str}"
            )
        print()

    print("== Answer ==\n")
    answer = synthesize(args.question, hits)
    print(answer.text)

    # Tail the sources so the user can click back to Discord.
    # Only show the chunks Claude actually cited, in order of first appearance.
    cited: list[str] = []
    seen: set[str] = set()
    for token in answer.text.split("[chunk:"):
        if "]" not in token:
            continue
        cid = token.split("]", 1)[0].strip()
        if cid and cid not in seen:
            seen.add(cid)
            cited.append(cid)

    id_to_hit = {h.chunk_id: h for h in hits}
    if cited:
        print("\n== Sources ==")
        for i, cid in enumerate(cited, start=1):
            h = id_to_hit.get(cid)
            if h is None:
                continue
            start = h.start_ts.strftime("%Y-%m-%d %H:%M") if h.start_ts else "?"
            url = _discord_jump_url(_resolve_channel_id(conn, cid), _resolve_start_message_id(conn, cid))
            print(f"  [{i}] chunk:{cid}  #{h.channel_name}  {start}")
            print(f"      {url}")

    print(
        f"\n-- {answer.model} | in={answer.input_tokens} out={answer.output_tokens} "
        f"cache_read={answer.cache_read_tokens} cache_write={answer.cache_creation_tokens}"
    )
    conn.close()


def _resolve_channel_id(conn, chunk_id: str) -> str:
    row = conn.execute(
        "SELECT channel_id FROM chunks WHERE chunk_id = ?", [chunk_id]
    ).fetchone()
    return row[0] if row else ""


def _resolve_start_message_id(conn, chunk_id: str) -> str:
    row = conn.execute(
        "SELECT start_message_id FROM chunks WHERE chunk_id = ?", [chunk_id]
    ).fetchone()
    return row[0] if row else ""


if __name__ == "__main__":
    main()

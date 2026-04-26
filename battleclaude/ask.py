"""CLI: retrieve + synthesize an answer for a natural-language question.

    python -m battleclaude.ask "What do 12lb horizontals range in for weapon MOI?"
"""
from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

CHUNK_MARKER_RE = re.compile(r"\[chunk:([A-Za-z0-9_\-]+)\]")


def _force_utf8_stdout() -> None:
    """Discord content contains emoji; Windows' default cp1252 console crashes on them."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

from .bm25 import DEFAULT_BM25_PATH
from .db import DEFAULT_DB_PATH, connect, resolve_chunk_jump_urls
from .retrieve import retrieve
from .synthesize import synthesize


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

    answer = synthesize(args.question, hits)

    # Collect cited chunk_ids in first-appearance order, then rewrite [chunk:id]
    # markers to [N] footnotes so the inline references match the Sources list.
    id_to_hit = {h.chunk_id: h for h in hits}
    cited: list[str] = []
    seen: set[str] = set()
    for m in CHUNK_MARKER_RE.finditer(answer.text):
        cid = m.group(1)
        if cid not in seen and cid in id_to_hit:
            seen.add(cid)
            cited.append(cid)
    cid_to_num = {cid: i + 1 for i, cid in enumerate(cited)}

    def _to_footnote(m: re.Match) -> str:
        n = cid_to_num.get(m.group(1))
        return f"[{n}]" if n else ""

    rendered = CHUNK_MARKER_RE.sub(_to_footnote, answer.text).strip()

    print("== Answer ==\n")
    print(rendered)

    if cited:
        urls = resolve_chunk_jump_urls(conn, cited)
        print("\n== Sources ==")
        for cid in cited:
            n = cid_to_num[cid]
            h = id_to_hit[cid]
            start = h.start_ts.strftime("%Y-%m-%d %H:%M") if h.start_ts else "?"
            print(f"  [{n}] chunk:{cid}  #{h.channel_name}  {start}")
            url = urls.get(cid)
            if url:
                print(f"      {url}")

    print(
        f"\n-- {answer.model} | in={answer.input_tokens} out={answer.output_tokens} "
        f"cache_read={answer.cache_read_tokens} cache_write={answer.cache_creation_tokens}"
    )
    conn.close()


if __name__ == "__main__":
    main()

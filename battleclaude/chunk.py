"""Turn the `messages` table into retrieval-ready conversation `chunks`.

One chunk is a sequence of consecutive non-empty, non-bot messages from the
same channel, sized to roughly TARGET_TOKENS by approximate token count, with
OVERLAP_TOKENS of trailing messages carried into the next chunk so a topic
that straddles the seam still appears in at least one chunk.

Chunk boundaries always sit at message boundaries — we never split a single
message across chunks, because that reads badly for both the embedder and
Claude.

Each chunk carries a `session_id` — all chunks from the same contiguous
conversation share that id so voyage-context-3 can embed each chunk with
awareness of its neighbours. Sessions break on gaps > SESSION_GAP.

Token counting here uses a fast char/4 heuristic. That's accurate enough for
chunk sizing; embed.py uses Voyage's real tokenizer for API-limit sizing.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from .db import connect, init_schema

SESSION_GAP = timedelta(hours=2)
TARGET_TOKENS = 600        # rough target size of each emitted chunk
OVERLAP_TOKENS = 100       # how much context to carry from chunk N into chunk N+1
CHARS_PER_TOKEN = 4        # heuristic — good enough for sizing, not for billing

CHUNK_INSERT_COLUMNS = [
    "chunk_id", "channel_id", "channel_name", "session_id",
    "start_message_id", "end_message_id",
    "start_timestamp", "end_timestamp",
    "message_count", "text",
]


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _format_chunk_text(
    messages: list[tuple],
    channel_name: str,
    start_ts: datetime,
    end_ts: datetime,
) -> str:
    if start_ts.date() == end_ts.date():
        date_range = start_ts.strftime("%Y-%m-%d %H:%M")
    else:
        date_range = f"{start_ts.strftime('%Y-%m-%d %H:%M')} - {end_ts.strftime('%Y-%m-%d %H:%M')}"
    lines = [f"[{channel_name} | {date_range}]"]
    for _mid, _ts, content, who in messages:
        if not content:
            continue
        content_clean = " ".join(content.split())
        lines.append(f"{who}: {content_clean}")
    return "\n".join(lines)


def _iter_sessions(messages: list[tuple]) -> Iterator[list[tuple]]:
    session: list[tuple] = []
    prev_ts: datetime | None = None
    for m in messages:
        ts = m[1]
        if prev_ts is not None and ts - prev_ts > SESSION_GAP:
            if session:
                yield session
            session = []
        session.append(m)
        prev_ts = ts
    if session:
        yield session


def _chunks_in_session(
    session: list[tuple],
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> Iterator[list[tuple]]:
    """Pack a session into ~target_tokens chunks with overlap at message boundaries.

    Pre-tokenize each message once with the char/4 heuristic. Walk forward,
    emitting a chunk once accumulated tokens cross `target_tokens`, then rewind
    the start pointer until `overlap_tokens` of trailing messages will repeat
    in the next chunk. Guarantees forward progress even on pathologically
    short-message sessions.
    """
    if not session:
        return

    tokens = [_approx_tokens(m[2]) for m in session]
    n = len(session)
    i = 0
    while i < n:
        # Extend the end of the chunk until we've covered target_tokens or run
        # out of messages. We always include at least one message, even if its
        # own token count already exceeds target.
        j = i
        running = 0
        while j < n and (j == i or running + tokens[j] <= target_tokens):
            running += tokens[j]
            j += 1

        yield session[i:j]

        if j >= n:
            break

        # Rewind the start pointer to give the next chunk `overlap_tokens` of
        # context. Always advance at least one message to avoid infinite loops
        # when a single message is larger than target_tokens.
        back = j
        back_running = 0
        while back > i + 1 and back_running < overlap_tokens:
            back -= 1
            back_running += tokens[back]
        i = max(back, i + 1)


def build_chunks(conn: duckdb.DuckDBPyConnection) -> int:
    # Drop-and-recreate so schema shape is always current (see db.BASE_SCHEMA_SQL).
    conn.execute("DROP TABLE IF EXISTS chunks")
    init_schema(conn)

    channels = conn.execute(
        "SELECT channel_id, name FROM channels ORDER BY channel_id"
    ).fetchall()

    all_rows: list[tuple] = []
    chunk_seq = 0
    for cid, cname in channels:
        messages = conn.execute(
            """
            SELECT message_id, timestamp, content,
                   COALESCE(author_nickname, author_name) AS who
            FROM messages
            WHERE channel_id = ?
              AND COALESCE(content, '') != ''
              AND COALESCE(author_is_bot, FALSE) = FALSE
            ORDER BY timestamp
            """,
            [cid],
        ).fetchall()

        channel_chunks = 0
        channel_sessions = 0
        for session_idx, session in enumerate(_iter_sessions(messages)):
            session_id = f"{cid}_s{session_idx:05d}"
            channel_sessions += 1
            for window in _chunks_in_session(session):
                start_ts = window[0][1]
                end_ts = window[-1][1]
                text = _format_chunk_text(window, cname, start_ts, end_ts)
                all_rows.append(
                    (
                        f"c{chunk_seq:08d}",
                        cid,
                        cname,
                        session_id,
                        window[0][0],
                        window[-1][0],
                        start_ts,
                        end_ts,
                        len(window),
                        text,
                    )
                )
                chunk_seq += 1
                channel_chunks += 1
        print(
            f"  {cname[:40]:<40}  {len(messages):>7,} msg -> "
            f"{channel_sessions:>5,} sessions, {channel_chunks:>6,} chunks"
        )

    df = pd.DataFrame(all_rows, columns=CHUNK_INSERT_COLUMNS)
    conn.register("_chunk_batch", df)
    try:
        column_list = ", ".join(CHUNK_INSERT_COLUMNS)
        conn.execute(
            f"INSERT INTO chunks ({column_list}) SELECT * FROM _chunk_batch"
        )
    finally:
        conn.unregister("_chunk_batch")

    total = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    sessions = int(conn.execute("SELECT COUNT(DISTINCT session_id) FROM chunks").fetchone()[0])

    # Spot-check chunk size distribution so tuning TARGET_TOKENS is easy.
    sizes = conn.execute(
        """
        SELECT
            MIN(LENGTH(text)) AS min_chars,
            AVG(LENGTH(text))::INT AS avg_chars,
            MAX(LENGTH(text)) AS max_chars,
            MEDIAN(LENGTH(text))::INT AS median_chars
        FROM chunks
        """
    ).fetchone()
    print(
        f"Built {total:,} chunks across {sessions:,} sessions "
        f"(chars: min={sizes[0]} median={sizes[3]} avg={sizes[1]} max={sizes[2]}, "
        f"~tokens = chars/{CHARS_PER_TOKEN})"
    )
    return total


def main() -> None:
    from argparse import ArgumentParser
    from .db import DEFAULT_DB_PATH

    parser = ArgumentParser(description="Build conversation chunks from the messages table.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    conn = connect(args.db)
    build_chunks(conn)
    conn.close()


if __name__ == "__main__":
    main()

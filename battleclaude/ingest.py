"""Stream DiscordChatExporter JSON files into DuckDB.

Usage:
    python -m battleclaude.ingest                 # default: data/raw/ -> battleclaude.duckdb
    python -m battleclaude.ingest --data-dir X    # override input dir
    python -m battleclaude.ingest --db Y.duckdb   # override output db

Design notes:
- No primary keys or secondary indexes exist during the bulk phase; they're
  added once at the end in finalize_schema(). That avoids per-row index work,
  which dominated the first (slow) version of ingest.
- Commit every COMMIT_EVERY rows so DuckDB can flush its transaction buffer
  and we don't balloon memory on the 379 MB main-channel file.
- Each file is idempotent: we DELETE any existing rows for the channel before
  inserting, so re-running on an updated export cleanly replaces that channel.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import duckdb
import ijson
import pandas as pd
from tqdm import tqdm

from .db import DEFAULT_DB_PATH, connect, finalize_schema, init_schema

# Rows per bulk append. At this size the Arrow conversion dominates instead of
# per-row SQL overhead, and DuckDB ingests tens of thousands of rows per second.
BATCH_SIZE = 25_000

THREAD_TYPES = {"GuildPublicThread", "GuildPrivateThread", "GuildNewsThread"}

MESSAGE_COLUMNS = [
    "message_id", "channel_id", "author_id",
    "timestamp", "timestamp_edited",
    "type", "content",
    "reply_to_message_id", "reply_to_channel_id",
    "is_pinned",
    "author_name", "author_nickname", "author_is_bot", "author_roles",
    "reactions", "attachments", "mentions",
]


def parse_header(path: Path) -> dict:
    """Read the top-level guild / channel / exportedAt fields without loading messages."""
    header: dict = {}
    for key in ("guild", "channel", "exportedAt"):
        with open(path, "rb") as f:
            header[key] = next(ijson.items(f, key), None)
    return header


def iter_messages(path: Path) -> Iterator[dict]:
    with open(path, "rb") as f:
        yield from ijson.items(f, "messages.item")


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    # DiscordChatExporter emits ISO 8601 with offset, e.g. 2023-03-17T00:35:17.324-04:00
    # Python 3.11+ handles this natively.
    return datetime.fromisoformat(s)


def message_to_row(msg: dict, channel_id: str) -> tuple[Any, ...]:
    author = msg.get("author") or {}
    reference = msg.get("reference") or {}
    return (
        msg.get("id"),
        channel_id,
        author.get("id"),
        _parse_ts(msg.get("timestamp")),
        _parse_ts(msg.get("timestampEdited")),
        msg.get("type"),
        msg.get("content") or "",
        reference.get("messageId"),
        reference.get("channelId"),
        bool(msg.get("isPinned", False)),
        author.get("name"),
        author.get("nickname"),
        bool(author.get("isBot", False)),
        json.dumps(author.get("roles") or []),
        json.dumps(msg.get("reactions") or []),
        json.dumps(msg.get("attachments") or []),
        json.dumps(msg.get("mentions") or []),
    )


def ingest_file(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    header = parse_header(path)
    channel = header["channel"] or {}
    guild = header["guild"] or {}
    channel_id = channel.get("id")
    if channel_id is None:
        raise ValueError(f"{path.name}: no channel.id in header")

    ch_type = channel.get("type")
    parent_channel_id = channel.get("categoryId") if ch_type in THREAD_TYPES else None

    # Idempotent re-ingest: drop any existing rows for this channel.
    conn.execute("DELETE FROM messages WHERE channel_id = ?", [channel_id])
    conn.execute("DELETE FROM channels WHERE channel_id = ?", [channel_id])

    conn.execute(
        """
        INSERT INTO channels
        (channel_id, guild_id, guild_name, category_id, category, name, type,
         topic, parent_channel_id, source_file, exported_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            channel_id,
            guild.get("id"),
            guild.get("name"),
            channel.get("categoryId"),
            channel.get("category"),
            channel.get("name"),
            ch_type,
            channel.get("topic"),
            parent_channel_id,
            path.name,
            _parse_ts(header["exportedAt"]) if isinstance(header["exportedAt"], str) else header["exportedAt"],
        ],
    )

    count = 0
    batch: list[tuple] = []
    label = path.name if len(path.name) <= 55 else path.name[:52] + "..."
    pbar = tqdm(desc=f"  {label}", unit=" msg", unit_scale=True, leave=True)

    def flush(rows: list[tuple]) -> None:
        df = pd.DataFrame.from_records(rows, columns=MESSAGE_COLUMNS)
        # Register as a view and INSERT SELECT — this uses DuckDB's zero-copy
        # Arrow path. `conn.append(table, df)` exists too but is stricter about
        # column types matching exactly; INSERT SELECT lets DuckDB cast.
        conn.register("_ingest_batch", df)
        try:
            conn.execute("INSERT INTO messages SELECT * FROM _ingest_batch")
        finally:
            conn.unregister("_ingest_batch")

    try:
        for msg in iter_messages(path):
            batch.append(message_to_row(msg, channel_id))
            count += 1
            if len(batch) >= BATCH_SIZE:
                flush(batch)
                pbar.update(len(batch))
                batch = []
        if batch:
            flush(batch)
            pbar.update(len(batch))
    finally:
        pbar.close()

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest DiscordChatExporter JSON into DuckDB.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--glob", default="*.json", help="glob within --data-dir (default: *.json)")
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="skip creating indexes after ingest (for incremental runs)",
    )
    args = parser.parse_args()

    json_files = sorted(args.data_dir.glob(args.glob))
    if not json_files:
        raise SystemExit(f"No files matching {args.glob!r} in {args.data_dir}")

    conn = connect(args.db)
    init_schema(conn)

    print(f"Ingesting {len(json_files)} file(s) into {args.db}")
    total = 0
    for path in json_files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name}  ({size_mb:,.1f} MB)")
        count = ingest_file(conn, path)
        total += count
        print(f"    -> {count:,} messages")

    if not args.no_finalize:
        print("Finalizing schema (creating indexes)...")
        finalize_schema(conn)

    stats = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM channels) AS channels,
            (SELECT COUNT(*) FROM messages) AS messages,
            (SELECT COUNT(DISTINCT author_id) FROM messages) AS authors,
            (SELECT MIN(timestamp) FROM messages) AS first_ts,
            (SELECT MAX(timestamp) FROM messages) AS last_ts
        """
    ).fetchone()
    conn.close()

    print()
    print(f"Done. {stats[1]:,} messages across {stats[0]} channels")
    print(f"      {stats[2]:,} distinct authors")
    print(f"      {stats[3]}  ->  {stats[4]}")


if __name__ == "__main__":
    main()

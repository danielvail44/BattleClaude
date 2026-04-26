from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path("battleclaude.duckdb")

# Base schema — NO primary keys, NO indexes. Those get added in finalize_schema()
# after bulk load so per-row index maintenance doesn't dominate ingest time.
BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS channels (
    channel_id          VARCHAR,
    guild_id            VARCHAR,
    guild_name          VARCHAR,
    category_id         VARCHAR,
    category            VARCHAR,
    name                VARCHAR,
    type                VARCHAR,
    topic               VARCHAR,
    parent_channel_id   VARCHAR,
    source_file         VARCHAR,
    exported_at         TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS messages (
    message_id              VARCHAR,
    channel_id              VARCHAR,
    author_id               VARCHAR,
    timestamp               TIMESTAMPTZ,
    timestamp_edited        TIMESTAMPTZ,
    type                    VARCHAR,
    content                 TEXT,
    reply_to_message_id     VARCHAR,
    reply_to_channel_id     VARCHAR,
    is_pinned               BOOLEAN,
    author_name             VARCHAR,
    author_nickname         VARCHAR,
    author_is_bot           BOOLEAN,
    author_roles            JSON,
    reactions               JSON,
    attachments             JSON,
    mentions                JSON
);

-- Retrieval layer: one row per conversation window.
-- `session_id` groups chunks that share conversational context, so
-- voyage-context-3 can embed each chunk conditioned on its neighbors.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id                VARCHAR,
    channel_id              VARCHAR,
    channel_name            VARCHAR,
    session_id              VARCHAR,
    start_message_id        VARCHAR,
    end_message_id          VARCHAR,
    start_timestamp         TIMESTAMPTZ,
    end_timestamp           TIMESTAMPTZ,
    message_count           INTEGER,
    text                    TEXT,
    embedding               FLOAT[]
);
"""

# Run once after bulk ingest. Indexes and uniqueness constraints go here.
# DuckDB doesn't yet support ALTER TABLE ADD PRIMARY KEY on a populated table,
# so we enforce uniqueness with a UNIQUE index instead — same effective guarantee.
FINALIZE_SCHEMA_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS uniq_channels_channel_id  ON channels(channel_id);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_messages_message_id  ON messages(message_id);
CREATE INDEX        IF NOT EXISTS idx_messages_channel_ts   ON messages(channel_id, timestamp);
CREATE INDEX        IF NOT EXISTS idx_messages_author       ON messages(author_id);
CREATE INDEX        IF NOT EXISTS idx_messages_reply        ON messages(reply_to_message_id);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_chunks_chunk_id      ON chunks(chunk_id);
CREATE INDEX        IF NOT EXISTS idx_chunks_channel        ON chunks(channel_id);
CREATE INDEX        IF NOT EXISTS idx_chunks_session        ON chunks(session_id);
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path))


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(BASE_SCHEMA_SQL)


def finalize_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(FINALIZE_SCHEMA_SQL)


def resolve_chunk_jump_urls(
    conn: duckdb.DuckDBPyConnection, chunk_ids: list[str]
) -> dict[str, str]:
    """chunk_id -> Discord jump URL, joining chunks to channels for guild_id.

    Used by all three frontends (ask, chat, bot) so jump links work for any
    Discord export — not just NHRL — without hardcoding a guild ID.
    """
    if not chunk_ids:
        return {}
    placeholders = ", ".join(["?"] * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT ch.chunk_id, c.guild_id, ch.channel_id, ch.start_message_id
        FROM chunks ch
        JOIN channels c USING (channel_id)
        WHERE ch.chunk_id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    out: dict[str, str] = {}
    for chunk_id, guild_id, channel_id, start_message_id in rows:
        if guild_id and channel_id and start_message_id:
            out[chunk_id] = (
                f"https://discord.com/channels/{guild_id}/{channel_id}/{start_message_id}"
            )
    return out

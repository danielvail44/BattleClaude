"""Quick sanity-check queries against the ingested DuckDB.

Usage: python scripts/sanity_check.py
"""
from __future__ import annotations

from battleclaude.db import connect


def main() -> None:
    conn = connect()

    print("== Channels ==")
    rows = conn.execute(
        """
        SELECT c.name, c.type, COUNT(m.message_id) AS msgs,
               MIN(m.timestamp) AS first, MAX(m.timestamp) AS last
        FROM channels c
        LEFT JOIN messages m USING(channel_id)
        GROUP BY c.channel_id, c.name, c.type
        ORDER BY msgs DESC
        """
    ).fetchall()
    for name, ctype, msgs, first, last in rows:
        print(f"  {msgs:>8,}  {ctype:<22}  {name}")
        print(f"           {first}  ->  {last}")

    print("\n== Top 15 posters ==")
    rows = conn.execute(
        """
        SELECT COALESCE(author_nickname, author_name) AS who,
               COUNT(*) AS n
        FROM messages
        WHERE author_is_bot = FALSE
        GROUP BY who
        ORDER BY n DESC
        LIMIT 15
        """
    ).fetchall()
    for who, n in rows:
        print(f"  {n:>7,}  {who}")

    print("\n== Messages by year ==")
    rows = conn.execute(
        """
        SELECT EXTRACT(year FROM timestamp)::INT AS yr, COUNT(*) AS n
        FROM messages
        GROUP BY yr
        ORDER BY yr
        """
    ).fetchall()
    for yr, n in rows:
        print(f"  {yr}  {n:>7,}")

    print("\n== Content stats ==")
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                                 AS total,
            SUM(CASE WHEN content = '' THEN 1 ELSE 0 END)            AS empty,
            SUM(CASE WHEN reply_to_message_id IS NOT NULL THEN 1 ELSE 0 END) AS replies,
            SUM(json_array_length(attachments))                      AS attachments,
            AVG(LENGTH(content))::INT                                AS avg_chars,
            MAX(LENGTH(content))                                     AS max_chars
        FROM messages
        """
    ).fetchone()
    total, empty, replies, attachments, avg_chars, max_chars = row
    print(f"  total messages:         {total:>8,}")
    print(f"  empty content:          {empty:>8,}  ({100*empty/total:.1f}%)")
    print(f"  replies (reference set):{replies:>8,}  ({100*replies/total:.1f}%)")
    print(f"  attachments:            {attachments:>8,}")
    print(f"  avg content length:     {avg_chars:>8,} chars")
    print(f"  max content length:     {max_chars:>8,} chars")

    print("\n== Top role tags across all messages ==")
    # UNNEST has to go in FROM in DuckDB, so we expand the JSON role array there.
    rows = conn.execute(
        """
        SELECT role, COUNT(*) AS n
        FROM messages,
             UNNEST(json_extract_string(author_roles, '$[*].name')) AS t(role)
        WHERE author_roles IS NOT NULL
          AND author_roles != '[]'
        GROUP BY role
        ORDER BY n DESC
        LIMIT 15
        """
    ).fetchall()
    for role, n in rows:
        print(f"  {n:>7,}  {role}")

    conn.close()


if __name__ == "__main__":
    main()

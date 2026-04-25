"""Interactive chat with follow-up question support.

Each turn:
  1. If history is non-empty, rewrite the user's input into a standalone
     search query (Haiku 4.5 — cheap and fast). Resolves referents like
     "tell me more about that ESC" -> "What do people say about Wraith 32?"
  2. Retrieve fresh chunks for the rewritten query.
  3. Add new chunks to the accumulated pool (deduped by chunk_id, capped).
  4. Synthesize with the full conversation history + the current pool.
  5. Append the turn to history.

Meta-commands:
  :quit / :exit / :q       leave
  :help                    show commands
  :history                 print prior turns
  :chunks                  print accumulated chunk pool
  :reset                   clear history + chunk pool (start fresh)
"""
from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .bm25 import DEFAULT_BM25_PATH
from .db import DEFAULT_DB_PATH, connect
from .retrieve import Hit, retrieve
from .synthesize import synthesize

REWRITE_MODEL = "claude-haiku-4-5"
MAX_POOL_CHUNKS = 60        # cap accumulated context bloat across turns
MAX_HISTORY_TURNS_FOR_REWRITE = 3  # how much history to feed the rewriter
PROMPT = "> "

NHRL_GUILD_ID = "651601084019900483"

REWRITE_SYSTEM = """\
You rewrite follow-up questions into standalone search queries for a
combat-robotics RAG system that searches a Discord corpus.

Given a short conversation history and a new user message, output a single
standalone search query that captures the user's current intent with all
referents resolved (e.g. "that ESC" -> "Wraith 32 ESC", "what about for \
30lbers" -> "30lb weapon MOI").

Output ONLY the rewritten query. No preamble, no quotes, no explanation. If
the new message is already standalone, output it verbatim.
"""


@dataclass
class Turn:
    question: str
    rewritten_query: str
    answer: str
    new_chunk_ids: list[str]


@dataclass
class ChatState:
    turns: list[Turn] = field(default_factory=list)
    # Insertion-ordered: oldest chunks at the front, newest at the back.
    pool: dict[str, Hit] = field(default_factory=dict)


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _discord_jump_url(channel_id: str, message_id: str) -> str:
    return f"https://discord.com/channels/{NHRL_GUILD_ID}/{channel_id}/{message_id}"


def _rewrite_question(client: anthropic.Anthropic, history: list[Turn], question: str) -> str:
    """Return a standalone search query for `question`, resolving prior context."""
    if not history:
        return question
    recent = history[-MAX_HISTORY_TURNS_FOR_REWRITE:]
    history_text = "\n\n".join(
        f"User: {t.question}\nAssistant: {t.answer[:400]}"
        + ("..." if len(t.answer) > 400 else "")
        for t in recent
    )
    user_msg = (
        f"CONVERSATION:\n{history_text}\n\n"
        f"NEW MESSAGE: {question}\n\n"
        f"Standalone query:"
    )
    resp = client.messages.create(
        model=REWRITE_MODEL,
        max_tokens=200,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text or question


def _add_to_pool(pool: dict[str, Hit], hits: list[Hit]) -> None:
    """Insert hits into pool (newest wins), then evict oldest to stay under the cap."""
    for h in hits:
        # Re-insert moves the entry to the end (most-recent).
        pool.pop(h.chunk_id, None)
        pool[h.chunk_id] = h
    while len(pool) > MAX_POOL_CHUNKS:
        # Pop oldest (insertion-order = first key).
        oldest = next(iter(pool))
        pool.pop(oldest)


def _print_sources(answer_text: str, pool: dict[str, Hit]) -> None:
    """Resolve [chunk:id] markers in answer_text to Discord jump links."""
    cited: list[str] = []
    seen: set[str] = set()
    for token in answer_text.split("[chunk:"):
        if "]" not in token:
            continue
        cid = token.split("]", 1)[0].strip()
        if cid and cid not in seen and cid in pool:
            seen.add(cid)
            cited.append(cid)
    if not cited:
        return
    print("\n  sources:")
    for i, cid in enumerate(cited, 1):
        h = pool[cid]
        start = h.start_ts.strftime("%Y-%m-%d %H:%M") if h.start_ts else "?"
        # Pull channel_id + start_message_id off the chunk row directly. We
        # don't have channel_id on Hit, so we reach via the Discord URL pieces
        # we DO have on the row. Simpler: refetch from DB if needed — but for
        # now we only have channel_name. Skip URL when we lack channel_id.
        print(f"  [{i}] chunk:{cid}  #{h.channel_name}  {start}")


def main() -> None:
    _force_utf8_stdout()
    load_dotenv()

    parser = ArgumentParser(description="Interactive chat with BattleClaude.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25_PATH)
    parser.add_argument("--top-k", type=int, default=30, help="chunks retrieved per turn")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="skip Voyage rerank (use raw RRF fusion order)",
    )
    parser.add_argument(
        "--show-rewrite",
        action="store_true",
        help="print the rewritten query before each retrieval",
    )
    args = parser.parse_args()

    conn = connect(args.db)
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    state = ChatState()

    print(
        "BattleClaude chat. Type your question, or :help for commands. "
        "Ctrl-D / Ctrl-C to exit.\n"
    )

    try:
        while True:
            try:
                question = input(PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue

            # Meta-commands
            if question in (":quit", ":exit", ":q"):
                break
            if question == ":help":
                print(
                    ":quit | :exit | :q   leave\n"
                    ":history             print prior turns\n"
                    ":chunks              print accumulated chunk pool\n"
                    ":reset               clear history + chunk pool"
                )
                continue
            if question == ":history":
                if not state.turns:
                    print("(no turns yet)")
                else:
                    for i, t in enumerate(state.turns, 1):
                        rw = "" if t.rewritten_query == t.question else f"  -> {t.rewritten_query!r}"
                        print(f"\n[{i}] Q: {t.question}{rw}")
                        snippet = t.answer if len(t.answer) <= 240 else t.answer[:237] + "..."
                        print(f"    A: {snippet}")
                continue
            if question == ":chunks":
                print(f"{len(state.pool)} chunks in pool (newest at bottom):")
                for cid, h in state.pool.items():
                    start = h.start_ts.strftime("%Y-%m-%d %H:%M") if h.start_ts else "?"
                    print(f"  {cid}  #{h.channel_name}  {start}")
                continue
            if question == ":reset":
                state = ChatState()
                print("(state cleared)")
                continue

            # Real turn
            rewritten = _rewrite_question(client, state.turns, question)
            if args.show_rewrite and rewritten != question:
                print(f"  search: {rewritten}\n")

            hits = retrieve(
                conn,
                rewritten,
                top_k=args.top_k,
                bm25_path=args.bm25,
                use_rerank=not args.no_rerank,
            )
            _add_to_pool(state.pool, hits)

            # History for synthesis: prior (question, answer) pairs.
            history = [(t.question, t.answer) for t in state.turns]

            # Pass the FULL accumulated pool as hits so Claude can re-cite
            # earlier evidence. Most-recently-added chunks end up last in the
            # pool, which is fine — the model attends across all of them.
            pool_hits = list(state.pool.values())
            answer = synthesize(question, pool_hits, history=history, client=client)

            print(answer.text)
            _print_sources(answer.text, state.pool)
            print(
                f"\n  -- {answer.model} | in={answer.input_tokens} "
                f"out={answer.output_tokens} cache_read={answer.cache_read_tokens} "
                f"| pool={len(state.pool)} chunks\n"
            )

            state.turns.append(
                Turn(
                    question=question,
                    rewritten_query=rewritten,
                    answer=answer.text,
                    new_chunk_ids=[h.chunk_id for h in hits],
                )
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

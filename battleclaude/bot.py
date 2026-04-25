"""Discord bot front-end for BattleClaude.

Usage model:
  /ask <question>            -> bot replies in-channel and opens a thread on
                                its reply. Subsequent messages in that thread
                                are treated as follow-ups (same retrieve +
                                rewrite + synthesize loop as `chat.py`).

State:
  In-memory dict keyed by thread_id -> ChatState. If the bot restarts, state
  is lost; on the next message in an unknown thread we rehydrate Q/A history
  from the thread's message log (chunk pool starts empty and refills naturally).

Concurrency:
  Retrieval (rewrite + vector matrix + BM25 + Voyage rerank) is synchronous
  and touches the shared duckdb connection. We run it via `asyncio.to_thread`
  under a single asyncio.Lock. Synthesis is run async via `AsyncAnthropic`
  with streaming, *outside* the lock — so a long Opus answer doesn't block
  the next user's retrieval.

Live responses:
  We use Anthropic's streaming API and edit Discord messages as deltas
  arrive, throttled to one edit per ~1.5s per message (well under Discord's
  edit rate limit). When the running answer crosses 2000 chars we spawn a new
  message and continue streaming into it. On stream completion we do a final
  edit pass that substitutes [chunk:id] markers for [N] footnotes plus a
  Sources block.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

import anthropic
import discord
from discord import app_commands
from dotenv import load_dotenv

from .bm25 import DEFAULT_BM25_PATH
from .chat import ChatState, Turn, _add_to_pool, _rewrite_question
from .db import DEFAULT_DB_PATH, connect
from .retrieve import retrieve
from .synthesize import (
    MAX_OUTPUT_TOKENS,
    MODEL,
    SYSTEM_PROMPT,
    format_chunks,
)

NHRL_GUILD_ID = "651601084019900483"
DISCORD_MSG_LIMIT = 2000
STREAM_SOFT_LIMIT = 1900       # leave headroom for a trailing cursor mark
STREAM_EDIT_INTERVAL = 1.5     # seconds between Discord edits while streaming
THREAD_NAME_LIMIT = 90
TOP_K = 30
USE_RERANK = True
REHYDRATE_LOOKBACK = 50  # how many recent thread messages to scan for history
STREAM_CURSOR = " ▌"

CHUNK_MARKER_RE = re.compile(r"\[chunk:([A-Za-z0-9_\-]+)\]")

log = logging.getLogger("battleclaude.bot")


def _split_for_discord(text: str, limit: int = DISCORD_MSG_LIMIT) -> list[str]:
    """Split text into <=limit chunks, preferring paragraph then line boundaries."""
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = remaining.rfind("\n", 0, limit)
        if cut == -1:
            cut = remaining.rfind(" ", 0, limit)
        if cut == -1 or cut < limit // 2:
            cut = limit
        out.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        out.append(remaining)
    return out


def _resolve_chunk_locations(conn, chunk_ids: list[str]) -> dict[str, tuple[str, str]]:
    """chunk_id -> (channel_id, start_message_id) for jump-link construction."""
    if not chunk_ids:
        return {}
    placeholders = ", ".join(["?"] * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, channel_id, start_message_id FROM chunks "
        f"WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def _format_answer_with_sources(answer_text: str, pool: dict, conn) -> str:
    """Replace [chunk:id] markers with [N] footnotes and append a sources block."""
    cited: list[str] = []
    seen: set[str] = set()
    for m in CHUNK_MARKER_RE.finditer(answer_text):
        cid = m.group(1)
        if cid not in seen and cid in pool:
            seen.add(cid)
            cited.append(cid)

    if not cited:
        return CHUNK_MARKER_RE.sub("", answer_text).strip()

    cid_to_num = {cid: i + 1 for i, cid in enumerate(cited)}

    def _replace(m: re.Match) -> str:
        cid = m.group(1)
        n = cid_to_num.get(cid)
        return f"[{n}]" if n else ""

    rewritten = CHUNK_MARKER_RE.sub(_replace, answer_text).strip()

    locations = _resolve_chunk_locations(conn, cited)
    src_lines = ["", "**Sources:**"]
    for cid in cited:
        n = cid_to_num[cid]
        h = pool[cid]
        start = h.start_ts.strftime("%Y-%m-%d %H:%M") if h.start_ts else "?"
        loc = locations.get(cid)
        if loc and loc[0] and loc[1]:
            url = f"https://discord.com/channels/{NHRL_GUILD_ID}/{loc[0]}/{loc[1]}"
            src_lines.append(f"[{n}] [#{h.channel_name} · {start}](<{url}>)")
        else:
            src_lines.append(f"[{n}] #{h.channel_name} · {start}")
    return rewritten + "\n" + "\n".join(src_lines)


async def _keep_typing(thread: discord.Thread) -> None:
    """Re-trigger Discord's 'is typing...' indicator every ~8s until cancelled.

    A single `thread.typing()` call expires after ~10s, so we loop. Used to
    show activity during retrieval + Opus's adaptive-thinking phase, before
    any text deltas have arrived to drive message edits.
    """
    try:
        while True:
            await thread.typing()
            await asyncio.sleep(8)
    except asyncio.CancelledError:
        return


class BattleClaudeBot(discord.Client):
    def __init__(self, db_path: Path, bm25_path: Path):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.db_path = db_path
        self.bm25_path = bm25_path
        self.conn = connect(db_path)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic = anthropic.Anthropic(api_key=api_key)
        self.async_anthropic = anthropic.AsyncAnthropic(api_key=api_key)
        self.states: dict[int, ChatState] = {}
        self.retrieval_lock = asyncio.Lock()

    async def setup_hook(self) -> None:
        await self.tree.sync()
        log.info("slash commands synced")

    def _prepare_turn(self, state: ChatState, question: str):
        """Sync: rewrite the question, retrieve, fold new chunks into the pool.

        Returns (rewritten_query, new_hits, pool_hits, history) — everything
        the streaming synthesis step needs. State.turns is NOT mutated here;
        the caller appends a Turn after the streaming completes successfully
        so a mid-stream failure leaves history coherent.
        """
        rewritten = _rewrite_question(self.anthropic, state.turns, question)
        hits = retrieve(
            self.conn,
            rewritten,
            top_k=TOP_K,
            bm25_path=self.bm25_path,
            use_rerank=USE_RERANK,
        )
        _add_to_pool(state.pool, hits)
        history = [(t.question, t.answer) for t in state.turns]
        pool_hits = list(state.pool.values())
        return rewritten, hits, pool_hits, history

    async def run_turn_streaming(
        self, thread: discord.Thread, state: ChatState, question: str
    ) -> None:
        """Full turn: prepare under lock, then stream synthesis into `thread`.

        A typing keepalive runs from entry until the first streamed text token
        arrives, so users see "BattleClaude is typing..." through the entire
        retrieval + Opus thinking phase (often 10-30s before any text emits).
        """
        typing_task = asyncio.create_task(_keep_typing(thread))
        try:
            async with self.retrieval_lock:
                rewritten, hits, pool_hits, history = await asyncio.to_thread(
                    self._prepare_turn, state, question
                )
            final_text = await self._stream_synthesis(
                thread, state.pool, question, pool_hits, history, typing_task
            )
        finally:
            typing_task.cancel()
        state.turns.append(
            Turn(
                question=question,
                rewritten_query=rewritten,
                answer=final_text,
                new_chunk_ids=[h.chunk_id for h in hits],
            )
        )

    async def _stream_synthesis(
        self,
        thread: discord.Thread,
        pool: dict,
        question: str,
        pool_hits: list,
        history: list[tuple[str, str]],
        typing_task: asyncio.Task | None = None,
    ) -> str:
        """Stream Claude's answer into `thread` as live message edits.

        Returns the raw answer text (with [chunk:id] markers intact, for
        history). The thread sees the formatted version with [N] footnotes
        and a Sources block.
        """
        if not pool_hits:
            empty = "No relevant discussion found in the corpus for this question."
            await thread.send(empty)
            return empty

        context = format_chunks(pool_hits)
        user_content = (
            f"DISCORD EXCERPTS ({len(pool_hits)} chunks):\n\n"
            f"{context}\n\n"
            f"USER QUESTION:\n{question}"
        )
        messages: list[dict] = []
        for prior_q, prior_a in history:
            messages.append({"role": "user", "content": prior_q})
            messages.append({"role": "assistant", "content": prior_a})
        messages.append({"role": "user", "content": user_content})

        placeholder = await thread.send("_thinking…_")
        msg_chain: list[discord.Message] = [placeholder]
        rendered_parts: list[str] = [""]
        full_text = ""
        last_edit = 0.0

        async def render(*, force: bool) -> None:
            nonlocal last_edit
            now = asyncio.get_event_loop().time()
            if not force and now - last_edit < STREAM_EDIT_INTERVAL:
                return
            last_edit = now
            display = (full_text + STREAM_CURSOR) if not force else full_text
            await self._sync_message_chain(
                thread, msg_chain, rendered_parts, display, STREAM_SOFT_LIMIT
            )

        async with self.async_anthropic.messages.stream(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=messages,
        ) as stream:
            async for text_delta in stream.text_stream:
                if typing_task is not None and not typing_task.done():
                    typing_task.cancel()
                    typing_task = None
                full_text += text_delta
                await render(force=False)

        # Final pass: substitute [chunk:id] -> [N] + Sources block, and
        # re-render. Use the full DISCORD_MSG_LIMIT here since we no longer
        # need cursor headroom.
        formatted = _format_answer_with_sources(full_text, pool, self.conn)
        await self._sync_message_chain(
            thread, msg_chain, rendered_parts, formatted, DISCORD_MSG_LIMIT
        )
        return full_text

    async def _sync_message_chain(
        self,
        thread: discord.Thread,
        msg_chain: list[discord.Message],
        rendered_parts: list[str],
        text: str,
        limit: int,
    ) -> None:
        """Make the chain of Discord messages match `text` split at `limit`.

        Spawns new messages when the chain is too short, deletes excess
        trailing messages, and only edits messages whose content has actually
        changed since the previous render (rendered_parts is the cache).
        """
        parts = _split_for_discord(text, limit=limit) or [""]

        while len(msg_chain) < len(parts):
            new_msg = await thread.send("…")
            msg_chain.append(new_msg)
            rendered_parts.append("…")

        for extra in msg_chain[len(parts):]:
            try:
                await extra.delete()
            except discord.HTTPException:
                pass
        del msg_chain[len(parts):]
        del rendered_parts[len(parts):]

        for i, (m, p) in enumerate(zip(msg_chain, parts)):
            content = p or "…"
            if rendered_parts[i] == content:
                continue
            try:
                await m.edit(content=content)
                rendered_parts[i] = content
            except discord.HTTPException as e:
                log.debug("edit failed (will retry next render): %s", e)

    async def rehydrate_state(self, thread: discord.Thread) -> ChatState:
        """Reconstruct Q/A history from the thread's messages. Pool starts empty."""
        state = ChatState()
        try:
            messages = [m async for m in thread.history(limit=REHYDRATE_LOOKBACK, oldest_first=True)]
        except discord.HTTPException:
            return state
        pending_q: str | None = None
        for m in messages:
            if m.author.id == self.user.id:
                if pending_q is not None and m.content:
                    cleaned = CHUNK_MARKER_RE.sub("", m.content)
                    cleaned = cleaned.split("**Sources:**")[0].strip()
                    state.turns.append(
                        Turn(
                            question=pending_q,
                            rewritten_query=pending_q,
                            answer=cleaned,
                            new_chunk_ids=[],
                        )
                    )
                    pending_q = None
            else:
                if m.content.strip():
                    pending_q = m.content.strip()
        return state


def _is_our_thread(thread: discord.Thread, bot_user_id: int) -> bool:
    """A thread is ours if its starter message was authored by the bot."""
    starter = thread.starter_message
    if starter is not None:
        return starter.author.id == bot_user_id
    # owner_id is who created the thread; bot creates threads via create_thread.
    return thread.owner_id == bot_user_id


def build_bot(db_path: Path, bm25_path: Path) -> BattleClaudeBot:
    bot = BattleClaudeBot(db_path, bm25_path)

    @bot.tree.command(name="ask", description="Ask BattleClaude about combat robotics")
    @app_commands.describe(question="Your question (a thread will open for follow-ups)")
    async def ask_cmd(interaction: discord.Interaction, question: str) -> None:
        await interaction.response.defer(thinking=True)

        # Anchor the conversation in-channel; full answer streams into the
        # thread that branches off the anchor.
        anchor_text = f"**{interaction.user.display_name} asked:** {question}"
        anchor_msg = await interaction.followup.send(
            anchor_text[:DISCORD_MSG_LIMIT], wait=True
        )

        # Followup webhook messages don't carry guild info, so create_thread
        # rejects them. Re-fetch through the channel to get a real Message.
        try:
            anchor = await interaction.channel.fetch_message(anchor_msg.id)
            thread = await anchor.create_thread(
                name=question[:THREAD_NAME_LIMIT] or "BattleClaude",
                auto_archive_duration=10080,  # 7d (Discord max)
            )
        except discord.HTTPException as e:
            log.warning("thread creation failed: %s", e)
            await interaction.channel.send(
                "Couldn't open a thread; aborting."
            )
            return

        state = ChatState()
        bot.states[thread.id] = state
        try:
            await bot.run_turn_streaming(thread, state, question)
        except Exception as e:
            log.exception("ask failed")
            await thread.send(f"Error: {e}")

    @bot.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot:
            return
        if not isinstance(message.channel, discord.Thread):
            return
        thread = message.channel
        if not _is_our_thread(thread, bot.user.id):
            return
        question = message.content.strip()
        if not question:
            return

        state = bot.states.get(thread.id)
        if state is None:
            state = await bot.rehydrate_state(thread)
            bot.states[thread.id] = state

        try:
            await bot.run_turn_streaming(thread, state, question)
        except Exception as e:
            log.exception("follow-up failed")
            await thread.send(f"Error: {e}")

    @bot.event
    async def on_ready() -> None:
        log.info("logged in as %s (id=%s)", bot.user, bot.user.id)

    return bot


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit("DISCORD_BOT_TOKEN not set in environment / .env")

    bot = build_bot(DEFAULT_DB_PATH, DEFAULT_BM25_PATH)
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()

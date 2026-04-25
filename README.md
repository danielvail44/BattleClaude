# BattleClaude

Retrieval-Augmented Generation over the NHRL 12lbs Discord server (~154k
messages from ~880 builders, 2020–present). Answers cite back to the
original Discord messages.

## Pipeline

1. **Ingest** — DiscordChatExporter JSON streamed into DuckDB.
2. **Chunk** — ~600-token conversation windows, snapped to message boundaries; sessions break on >2-hour gaps.
3. **Embed** — Voyage `voyage-context-3` (1024-dim, contextualized per session).
4. **BM25** — keyword index alongside the vectors, with a jargon-preserving tokenizer (keeps `Wraith-32`, `VESC 6.0`, `V+`).
5. **Retrieve** — top-200 vector ∪ top-200 BM25, fused via Reciprocal Rank Fusion, then Voyage `rerank-2.5` cross-encoder narrows to top-30.
6. **Synthesize** — Claude Opus 4.7 with adaptive thinking, `[chunk:id]` citations resolved to Discord jump links.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .   # Linux/macOS: .venv/bin/pip

cp .env.example .env
# ANTHROPIC_API_KEY  — synthesis + query rewriting
# VOYAGE_API_KEY     — embeddings + reranker (free tier covers the corpus)
# DISCORD_BOT_TOKEN  — bot only
```

Drop DiscordChatExporter JSON in `data/raw/`, then build the indexes:

```bash
python -m battleclaude.ingest    # JSON -> DuckDB
python -m battleclaude.index     # chunk -> embed -> BM25
```

## Usage

```bash
# One-shot question
python -m battleclaude.ask "What weapon ESCs have people had success with?"

# Interactive REPL with follow-up support
python -m battleclaude.chat

# Discord bot — /ask opens a thread, follow-ups go in the thread
python -m battleclaude.bot
```

`ask` flags: `--show-chunks`, `--no-rerank`, `--top-k N`.
`chat` meta-commands: `:help`, `:history`, `:chunks`, `:reset`, `:quit`.

For the bot: create a Discord app, enable **Message Content Intent**, invite
with scopes `bot` + `applications.commands` and permissions Send Messages /
Create Public Threads / Send Messages in Threads / Read Message History.

## Models

- Embeddings: Voyage `voyage-context-3`
- Reranker: Voyage `rerank-2.5`
- Query rewriter (chat/bot): Claude Haiku 4.5
- Synthesis: Claude Opus 4.7 with adaptive thinking

## Repo layout

```
battleclaude/
  ingest.py       JSON -> DuckDB
  chunk.py        token-based chunking, session-aware
  embed.py        Voyage voyage-context-3, batched per session
  bm25.py         BM25 index
  retrieve.py     RRF fusion + cross-encoder rerank
  synthesize.py   Claude synthesis with [chunk:id] citations
  ask.py          one-shot CLI
  chat.py         interactive REPL
  bot.py          Discord bot with live streaming
  index.py        chunk + embed + BM25 orchestrator
scripts/
  sanity_check.py corpus statistics
```

[CLAUDE.md](CLAUDE.md) has the design rationale and limitations.

## Run as a service

```ini
# /etc/systemd/system/battleclaude.service
[Unit]
Description=BattleClaude Discord bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/BattleClaude
ExecStart=/home/pi/BattleClaude/.venv/bin/python -m battleclaude.bot
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now battleclaude
journalctl -u battleclaude -f
```

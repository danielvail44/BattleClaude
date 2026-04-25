# BattleClaude

Extract insights from the NHRL 12lbs Discord corpus (6+ years of combat-robotics
chat, ~153k messages). The pipeline is **ingest → chunk → embed → BM25 →
hybrid retrieve + rerank → Claude synthesis**, with a one-shot `ask` mode and
an interactive `chat` mode that supports follow-up questions.

## First-time setup

```bash
# 1. Virtualenv + install (once)
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .

# 2. API keys. Copy and fill in your keys:
cp .env.example .env        # then edit .env:
#   VOYAGE_API_KEY=...      https://www.voyageai.com         (embeddings + reranker)
#   ANTHROPIC_API_KEY=...   https://console.anthropic.com    (synthesis + query rewriting)

# 3. Ingest the Discord JSON drops in data/raw/ into DuckDB (only needed once
#    per data drop; idempotent per channel).
python -m battleclaude.ingest

# 4. Build the retrieval indexes: chunks → Voyage embeddings → BM25.
#    Embeddings use voyage-context-3 (contextualized chunk embeddings, grouped
#    by session). Voyage gives 200M free tokens — our corpus is ~5M, so the
#    whole embedding step is free on a new account.
python -m battleclaude.index

# 5a. Ask one-shot questions.
python -m battleclaude.ask "What do 12lb horizontals range in for weapon MOI?"
python -m battleclaude.ask "What weapon ESCs have people had success with?"
python -m battleclaude.ask "Is FOC or trapezoidal the favored VESC mode?"

# 5b. Or open an interactive chat with follow-up support.
python -m battleclaude.chat

# 5c. Or run as a Discord bot (private server). Needs DISCORD_BOT_TOKEN in .env
#     and the "Message Content Intent" enabled on the bot. /ask <q> answers in
#     the channel and opens a thread; subsequent messages in that thread are
#     follow-ups (same retrieve+rewrite+synthesize loop as `chat`).
python -m battleclaude.bot
```

Useful flags for `ask`:
- `--show-chunks` — print retrieved chunks (with vec / bm25 / rrf / rerank ranks) before the answer.
- `--no-rerank` — skip the Voyage rerank stage (raw RRF order). Useful for A/B comparison.
- `--top-k 40` — widen context (default 30).

Useful flags for `chat`:
- `--show-rewrite` — print the standalone-query rewrite for each follow-up.
- `--no-rerank`, `--top-k N` — same as `ask`.
- Meta-commands inside the REPL: `:help`, `:history`, `:chunks`, `:reset`, `:quit`.

Useful flags for `index`:
- `--rechunk` — drop and rebuild chunks from messages (re-embed is required after).
- `--skip-embed` — chunks + BM25 only; handy for offline iteration on the BM25 path.
- `--reembed-all` — re-embed every chunk, not just `embedding IS NULL`.

## What's in the repo

- `data/raw/` — DiscordChatExporter JSON files. Gitignored.
- `battleclaude.duckdb` — the ingested corpus + chunks + embeddings. Gitignored.
- `battleclaude.bm25.pkl` — pickled BM25 index. Gitignored.
- `battleclaude/` — the package:
  - `db.py` — DuckDB connection + canonical `channels` / `messages` / `chunks` schema. Indexes are added in a separate `finalize_schema()` step after bulk load to avoid per-row index maintenance during ingest.
  - `ingest.py` — streams DiscordChatExporter JSON into DuckDB via pandas + `INSERT ... SELECT`. Idempotent per channel (DELETE-then-insert).
  - `chunk.py` — token-based conversation chunking (~600 tokens per chunk, 100-token overlap, snapped to message boundaries). Sessions break on >2-hour gaps; every chunk carries a `session_id` so `embed.py` can group it with its neighbours.
  - `embed.py` — Voyage `voyage-context-3` embedder (1024-dim, contextualized chunk embeddings). Batches chunks by `session_id` so each chunk is embedded aware of its conversational context. Splits oversized sessions, packs requests under per-request token/chunk/document caps. Exponential-backoff on transient errors; resumable (only embeds chunks where `embedding IS NULL`).
  - `bm25.py` — `rank_bm25` index with a jargon-preserving tokenizer (keeps `VESC`, `Wraith-32`, `V+`, etc.).
  - `retrieve.py` — two-stage hybrid retrieval. Stage 1 (recall): top-200 vector ∪ top-200 BM25, fused via Reciprocal Rank Fusion → top 100. Stage 2 (precision): Voyage `rerank-2.5` cross-encoder → top 30. Reranking can be skipped via `use_rerank=False`.
  - `rerank.py` — Voyage `rerank-2.5` client with exponential-backoff retry.
  - `synthesize.py` — Claude **Opus 4.7** call with `thinking: adaptive` and a combat-robotics system prompt. Accepts optional `history=[(question, answer), ...]` so multi-turn callers can pass prior turns. `cache_control` on the system block so caching activates the moment the prompt grows past the cache threshold.
  - `ask.py` — one-shot CLI: retrieve + synthesize + print answer with Discord jump-link sources.
  - `chat.py` — interactive REPL CLI with follow-up support. Each turn: rewrites the user's input into a standalone search query (Haiku 4.5, ~$0.001 per call) when there's prior history, retrieves fresh chunks, accumulates them in a FIFO chunk pool (cap 60), and synthesises with full message history.
  - `index.py` — orchestrator that runs chunk → embed → BM25 in one shot.
- `scripts/sanity_check.py` — quick DB-level statistics (counts, top authors, year distribution, role distribution).

## Pipeline shape

```
DiscordChatExporter JSON
        │  ingest.py (ijson stream → DuckDB via pandas)
        ▼
   messages table  (153k rows)
        │  chunk.py (~600-tok windows, 100-tok overlap, session-grouped)
        ▼
   chunks table    (7.7k rows, with session_id for context-aware embed)
        │  embed.py (voyage-context-3 per session)
        ▼
   chunks.embedding   +   battleclaude.bm25.pkl
                              │
                              │  retrieve.py
                              ▼
            vec top-200  ─┐
                          ├─ RRF fuse → top 100 → Voyage rerank-2.5 → top 30
            BM25 top-200 ─┘
                              │
                              ▼
        synthesize.py (Claude Opus 4.7, adaptive thinking)
                              │
                              ▼
                  answer + [chunk:id] citations + Discord jump links
```

`chat.py` wraps the retrieve→synthesise loop with conversation state and a
Haiku-4.5 query-rewriter for follow-ups.

## Corpus snapshot (2026-04-23 drop)

- **153,557 messages** across **9 channels**, **880 authors**, **2020-01-10 → 2026-04-23**
- **7,768 chunks** across **4,832 sessions** (2-hour session gap, ~600-token chunks with 100-token overlap, snapped to message boundaries)
- Chunk size distribution: median ~488 tokens, max ~1,220 tokens
- **77%** of messages are from authors with the `NHRL Competitor` role — strong signal that we're reading actual builders, not spectators.

## Models in use

- **Embeddings:** Voyage `voyage-context-3` (1024-dim, contextualized per session)
- **Reranker:** Voyage `rerank-2.5` (cross-encoder, top-100 → top-30)
- **Query rewriter (chat only):** Claude Haiku 4.5
- **Synthesis:** Claude Opus 4.7 with `thinking: adaptive`

## Known limitations / next steps

- Vector search is naive full-matrix cosine (~30 MB `np.float32` in memory).
  Fine at 7.7k chunks; switch to DuckDB `vss` / HNSW when the corpus grows.
- No entity resolution yet (same ESC called "Wraith 32", "WK32", "wraith" in
  different messages → separate surface forms). Claude handles it in synthesis
  reasonably but we lose clean aggregation.
- Vision on attachments (CAD renders, photos of bots) is deferred. The
  `attachments` JSON column on `messages` has the URLs and filenames; ~12k
  attachments are unindexed.
- No author-reputation prior in retrieval (e.g. boost messages from `NHRL
  Competitor` authors). All chunks weighted equally today.
- No date filtering on retrieval (e.g. "consensus in 2024 vs 2025").
- No eval harness — quality is judged by eyeballing answers. Building a small
  gold set would let us measure recall@k and tune systematically.
- `chat` sessions are in-memory only; no save/resume to disk yet.

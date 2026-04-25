"""Claude-based synthesis: take retrieved chunks, produce an answer with citations.

Design notes:
- Model: Claude Opus 4.7 — strongest reasoning available, used because answer
  quality (reconciling conflicting builder advice across years of chat) is the
  bottleneck, not throughput or cost. Drop to Sonnet 4.6 if iteration speed
  matters more than synthesis quality.
- Adaptive thinking is on so Claude can reconcile conflicting messages and
  reason about numeric ranges before writing.
- Prompt caching: `cache_control` sits on the last system block. Our system
  prompt is short today (well under the cache minimum) so it won't cache on
  day one — but this keeps caching automatic the moment the system prompt
  grows past the threshold or we add a frozen preamble.
- Citation format: Claude emits `[chunk:<id>]` markers inline. The CLI layer
  resolves those back to Discord jump links using the message IDs on each
  chunk row.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic

from .retrieve import Hit

MODEL = "claude-opus-4-7"
MAX_OUTPUT_TOKENS = 4096

SYSTEM_PROMPT = """\
You are BattleClaude, an assistant that answers questions about combat robotics \
using excerpts from the NHRL Discord server. The corpus is dominated by 12lb \
bot builders — expect jargon like VESC, FOC, trapezoidal, MOI, HDPE, UHMW, \
brushless, ESC, spinner, drum, vertical, horizontal, beetle, antweight.

You will be given a USER QUESTION and a set of DISCORD EXCERPTS. Each excerpt \
begins with a header of the form `[chunk:<id> | <channel> | <time-range>]` \
followed by a Discord transcript.

Rules for answering:
1. Base the answer only on the excerpts. Do NOT rely on outside knowledge about \
specific products, builders, or events — those claims belong in the excerpts.
2. Cite evidence inline with `[chunk:<id>]` markers. Every non-trivial claim \
needs at least one citation. You may cite multiple chunks for the same claim: \
`[chunk:abc123][chunk:def456]`.
3. For range / aggregate questions (e.g. "what do X range in..."), extract the \
specific numbers or entities mentioned, aggregate them, and report the range \
or consensus. Show the reader what you actually found, not a hedge.
4. If the excerpts disagree, say so and cite both sides.
5. If the excerpts don't actually answer the question, say that plainly. Don't \
paper over gaps with generic knowledge.
6. Keep the tone direct and technical. No preamble like "Based on the \
excerpts...". Just answer.
"""


@dataclass
class Answer:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int


def format_chunks(hits: list[Hit]) -> str:
    blocks: list[str] = []
    for h in hits:
        start = h.start_ts.strftime("%Y-%m-%d %H:%M") if h.start_ts else ""
        end = h.end_ts.strftime("%Y-%m-%d %H:%M") if h.end_ts else ""
        time_range = f"{start} - {end}" if start and end else (start or end or "?")
        header = f"[chunk:{h.chunk_id} | #{h.channel_name} | {time_range}]"
        blocks.append(f"{header}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


def synthesize(
    question: str,
    hits: list[Hit],
    *,
    history: list[tuple[str, str]] | None = None,
    client: anthropic.Anthropic | None = None,
) -> Answer:
    """Answer `question` over `hits`, optionally with prior `(question, answer)` history.

    When history is provided, prior turns are sent as user/assistant messages so
    Claude can interpret follow-ups in context. The current turn's chunks are
    placed in the latest user message; chunks from prior turns are NOT re-sent
    (the caller is expected to merge prior+new chunks into `hits` before this
    call if it wants past evidence to remain citable).
    """
    if not hits:
        return Answer(
            text="No relevant discussion found in the corpus for this question.",
            model=MODEL,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )

    client = client or anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    context = format_chunks(hits)
    user_content = (
        f"DISCORD EXCERPTS ({len(hits)} chunks):\n\n"
        f"{context}\n\n"
        f"USER QUESTION:\n{question}"
    )

    messages: list[dict] = []
    for prior_q, prior_a in history or []:
        messages.append({"role": "user", "content": prior_q})
        messages.append({"role": "assistant", "content": prior_a})
    messages.append({"role": "user", "content": user_content})

    response = client.messages.create(
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
    )

    text_parts = [b.text for b in response.content if b.type == "text"]
    answer_text = "\n\n".join(text_parts).strip() or "(empty response from model)"

    usage = response.usage
    return Answer(
        text=answer_text,
        model=response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )

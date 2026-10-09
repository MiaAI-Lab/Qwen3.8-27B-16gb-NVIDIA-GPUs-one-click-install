"""How many tokens a request may generate: pure arithmetic, standard library only.

ExLlamaV3 admits a job only if (prompt + max_new_tokens), rounded up to whole
cache pages, fits in the entire KV cache (generator/pagetable.py and
generator/job.py). A client that asks for max_tokens=65536 on a 91k-token prompt
in a 117k cache is therefore rejected outright ("Job requires 485 pages (only
460 available)") although the reply would have been a few hundred tokens.
Clamping max_new_tokens to what is left turns that into a normal answer that
ends with finish_reason=length when it really runs out of room.
"""
from __future__ import annotations

PAGE_SIZE = 256            # the engine's cache page, in tokens
DEFAULT_MAX_TOKENS = 32768  # used when the client sends no max_tokens


def usable_cache_tokens(cache_tokens: int) -> int:
    """Whole pages only: the engine cannot use a partial page."""
    return (int(cache_tokens) // PAGE_SIZE) * PAGE_SIZE


def clamp_max_tokens(prompt_tokens: int, max_tokens: int, cache_tokens: int | None) -> int:
    """max_tokens reduced so that the job fits the cache.

    Raises ValueError when the prompt alone does not fit, with a message the
    client can show (it is the signal to compact the conversation).
    An unknown cache size (None or 0) leaves the request untouched."""
    max_tokens = max(1, int(max_tokens))
    if not cache_tokens:
        return max_tokens
    usable = usable_cache_tokens(cache_tokens)
    room = usable - int(prompt_tokens) - 1      # -1: the engine counts one more
    if room < 1:
        raise ValueError(
            f"the prompt is {prompt_tokens} tokens but the KV cache holds {usable}; "
            "shorten the conversation (or raise CONTEXT_SIZE if the card has room)")
    return min(max_tokens, room)

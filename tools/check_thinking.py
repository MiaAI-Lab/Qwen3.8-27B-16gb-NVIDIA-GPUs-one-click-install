#!/usr/bin/env python3
"""Does the running server actually stream reasoning ("Thinking") to the UI?

    .venv\\Scripts\\python.exe tools\\check_thinking.py

Sends one prompt that a reasoning model should not be able to answer without
working through it, reads the raw SSE stream, and reports what arrived. It
touches nothing: no .env, no sessions, no model reload.

Why this exists: "no Thinking in the chat" has three completely different
causes and they need different fixes.
  * the model never opened a think block   -> prompt/template, not the UI
  * it thought but sent nothing to the UI  -> the server's split_reasoning
  * both arrived and nothing was drawn     -> the web UI
Only the raw stream can tell them apart.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
                  "than the ball. How much does the ball cost? Think it through.")


def env(name: str, default: str = "") -> str:
    path = ROOT / ".env"
    if not path.is_file():
        return default
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(name + "=") and not line.startswith("#"):
            return line.split("=", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
    return default


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=env("PORT", "8888"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--raw", action="store_true", help="print every delta as it arrives")
    a = ap.parse_args(argv)

    url = f"http://{a.host}:{a.port}/v1/chat/completions"
    body = json.dumps({
        "model": env("MODEL_ID", "local"),
        "messages": [{"role": "user", "content": a.prompt}],
        "stream": True,
        "max_tokens": a.max_tokens,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer local"})

    print(f"  asking {url}")
    print(f"  prompt: {a.prompt[:70]}...\n")
    reasoning, content, other = [], [], []
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                        if a.raw:
                            print("  [reasoning]", repr(delta["reasoning_content"][:60]))
                    elif delta.get("content"):
                        content.append(delta["content"])
                        if a.raw:
                            print("  [content]  ", repr(delta["content"][:60]))
                    elif delta:
                        other.append(delta)
    except urllib.error.URLError as e:
        print(f"  could not reach the server: {e}")
        print(f"  is Simplex running? the chat UI would be at http://{a.host}:{a.port}/")
        return 2

    r_text, c_text = "".join(reasoning), "".join(content)
    print(f"  reasoning deltas : {len(reasoning):>4}   {len(r_text)} chars")
    print(f"  content deltas   : {len(content):>4}   {len(c_text)} chars")
    if other:
        print(f"  other deltas     : {len(other):>4}   e.g. {other[0]}")
    print()
    if r_text:
        print("  Thinking (first 200 chars):")
        print("   ", r_text[:200].replace("\n", "\n    "))
    print("\n  Answer (first 200 chars):")
    print("   ", c_text[:200].replace("\n", "\n    "))
    print()

    # the marker leaking into the answer means the split missed it
    leaked = "</think>" in c_text or "<think>" in c_text
    if leaked:
        print("  ! The answer still contains a <think> marker, so the server did not")
        print("    split reasoning out of it. That is a server bug, not a UI one.")
        return 1
    if r_text:
        print("  OK  The server streamed reasoning. If the chat shows no Thinking")
        print("      section for this prompt, the problem is in the web UI.")
        return 0
    print("  !  No reasoning arrived. The model answered without opening a think")
    print("     block - for a simple prompt that is normal (hybrid reasoning models")
    print("     skip it), but for this one it suggests thinking is not being")
    print("     enabled in the prompt template. Re-run with --raw to see the stream.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

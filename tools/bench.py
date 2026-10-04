"""Decode/prefill benchmark against a running Simplex server (standard library only).

Talks to /v1/chat/completions exactly like a client does, so what it measures
includes MTP drafting, the chat template and the HTTP path - the numbers a
user sees, not a kernel microbenchmark.

    .venv/bin/python tools/bench.py                       # short + 64k + 128k + 240k
    .venv/bin/python tools/bench.py --contexts 0,200000 --runs 3 --json out.json

For each context size it pads the prompt with real source text up to about
that many tokens, then asks for --gen tokens. Reported per run:
    prefill  prompt tokens / time to the first streamed token (cold on the
             first run at each size only - later runs hit the prefix cache)
    decode   (completion tokens - 1) / time from first to last token
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK = ("Write a complete, well-commented Python implementation of a thread-safe "
        "LRU cache with TTL expiry, then a pytest test suite for it. "
        "Answer directly with the code.")


def filler(approx_tokens: int) -> str:
    """Real source text (this kit's own tools/) repeated to ~approx_tokens."""
    if approx_tokens <= 0:
        return ""
    src = "\n\n".join(p.read_text(errors = "replace")
                      for p in sorted((ROOT / "tools").glob("*.py")))
    want = int(approx_tokens * 3.3)       # ~3.3 chars/token for Python on Qwen
    out = (src * (want // len(src) + 1))[:want]
    # The size leads the prompt so no two sizes share a cached prefix: the
    # first run at each size measures a cold prefill, later runs reuse it.
    return (f"[{approx_tokens}] Here is some reference code. Read it, then do the task after it.\n\n"
            f"```python\n{out}\n```\n\n")


def run_one(url: str, model: str, ctx: int, gen: int, timeout: float,
            temperature: float | None = None) -> dict:
    body = {
        "model": model, "stream": True, "max_tokens": gen,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": filler(ctx) + TASK}],
    }
    if temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data = json.dumps(body).encode(),
                                 headers = {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = {}
    with urllib.request.urlopen(req, timeout = timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if any(d.get(k) for k in ("content", "reasoning_content", "reasoning")):
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
    p = int(usage.get("prompt_tokens", 0))
    c = int(usage.get("completion_tokens", 0))
    ttft = (t_first or t0) - t0
    dec_t = (t_last or t0) - (t_first or t0)
    return {"ctx_target": ctx, "prompt_tokens": p, "completion_tokens": c,
            "ttft_s": round(ttft, 2),
            "prefill_tps": round(p / ttft, 1) if ttft > 0 else None,
            "decode_tps": round((c - 1) / dec_t, 1) if dec_t > 0 and c > 1 else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default = "http://127.0.0.1:8888")
    ap.add_argument("--contexts", default = "0,65536,131072,240000",
                    help = "comma list of approximate prompt sizes in tokens")
    ap.add_argument("--gen", type = int, default = 768)
    ap.add_argument("--runs", type = int, default = 2)
    ap.add_argument("--timeout", type = float, default = 1800)
    ap.add_argument("--temperature", type = float, default = None,
                    help = "0 = greedy: the same tokens every run, so MTP acceptance "
                           "stops adding noise to A/B comparisons (default: server's 0.6)")
    ap.add_argument("--json", help = "append the results to this JSON-lines file")
    ap.add_argument("--label", default = "", help = "tag stored with each row")
    a = ap.parse_args()

    with urllib.request.urlopen(a.url + "/v1/models", timeout = 10) as r:
        model = json.load(r)["data"][0]["id"]
    run_one(a.url, model, 0, 32, a.timeout)                       # warm-up
    print(f"model {model}   gen {a.gen}   runs {a.runs}")
    print(f"{'target':>8} {'prompt':>8} {'out':>5} {'ttft s':>7} {'prefill t/s':>12} {'decode t/s':>11}")
    for ctx in (int(x) for x in a.contexts.split(",")):
        for _ in range(a.runs):
            row = run_one(a.url, model, ctx, a.gen, a.timeout, a.temperature)
            row["label"], row["model"], row["temperature"] = a.label, model, a.temperature
            print(f"{ctx:>8} {row['prompt_tokens']:>8} {row['completion_tokens']:>5} "
                  f"{row['ttft_s']:>7} {row['prefill_tps'] or '-':>12} {row['decode_tps'] or '-':>11}",
                  flush = True)
            if a.json:
                with open(a.json, "a") as f:
                    f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

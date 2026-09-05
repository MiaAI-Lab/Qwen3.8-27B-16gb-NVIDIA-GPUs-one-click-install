#!/usr/bin/env python3
"""KV-cache format quality + speed test on the real model (needs the GPU).

Compares cache formats against an fp16-cache baseline using teacher-forced
next-token distributions, a long-context passkey retrieval, decode speed and
VRAM use. Run from the kit root with the venv Python (test_kv.bat does it):

    .venv\\Scripts\\python.exe tools\\kv_cache_tests.py                 # everything
    .venv\\Scripts\\python.exe tools\\kv_cache_tests.py --modes fp16/4/nvfp4 --quick
    .venv\\Scripts\\python.exe tools\\kv_cache_tests.py --no-passkey

What is measured (per cache format, one model load each):
  KL          mean / p95 / max KL(fp16 || format) of the next-token distribution at
              N cut points inside the model's own qbench transcripts
              (models/.../qbench_prompts_gen.json: prompt + reference response)
  top-1       how often the format's argmax token equals fp16's argmax
  ref-acc     how often the argmax equals the reference transcript token (sanity)
  passkey     retrieval of a hidden key from a --passkey-len token filler (formats
              other than fp16; fp16 at that length does not fit 16 GB)
  decode      tokens/s for 96 greedy tokens after a 2k-token prompt
  VRAM        allocated / reserved after load (context = --context)

Reading the result: int '4' vs 'nvfp4' at the same 4.5 bits/elem is the decision
for the 150k profile; fp8 vs '8' is the 8-bit question. Lower KL is better; the
fp16 row is the noise floor of the measurement itself (it should be ~0).

fp8 / nvfp4 need tools/patch_kv.py applied (test_kv.bat applies it).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

ALL_MODES = ["fp16", "8", "8,4", "fp8", "4", "nvfp4"]
BITS = {"fp16": 16, "8": 8.5, "8,4": 6.5, "fp8": 8.0, "4": 4.5, "nvfp4": 4.5}


def load_env():
    from win_start import load_dotenv  # type: ignore
    return load_dotenv(ROOT / ".env")


def log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------- model --

def build(model_dir: str, gpu_mem: str, cache_size: int, mode: str):
    """Load model + cache of the requested format. Returns (model, cache, tokenizer, generator)."""
    from argparse import ArgumentParser
    from exllamav3 import model_init, Generator
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args=False)
    argv = ["-m", model_dir, "-gs", str(gpu_mem), "-cs", str(cache_size)]
    if mode != "fp16":
        argv += ["-cq", mode]
    args = parser.parse_args(argv)
    model, config, cache, tokenizer = model_init.init(args, progress=True)
    generator = Generator(model, cache, tokenizer)
    return model, cache, tokenizer, generator


def unload(model, cache, generator):
    import torch
    try:
        model.unload()
    except Exception:
        pass
    del generator, cache, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def vram_gib():
    import torch
    return torch.cuda.memory_allocated(0) / 1024 ** 3, torch.cuda.memory_reserved(0) / 1024 ** 3


# ------------------------------------------------------------------- probes ---

def next_token_logits(generator, tokenizer, ids):
    """Greedy 1-token job with return_logits -> (logits fp32 cpu [vocab], argmax)."""
    import torch
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import ArgmaxSampler
    input_ids = torch.tensor([ids], dtype=torch.long)
    job = Job(input_ids=input_ids, max_new_tokens=1, sampler=ArgmaxSampler(),
              return_logits=True, decode_special_tokens=True)
    generator.enqueue(job)
    logits = None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if "logits" in r and r["logits"] is not None and r["logits"].numel():
                logits = r["logits"][0, -1].float().cpu()
    if logits is None:
        raise RuntimeError("job returned no logits")
    return logits


def generate(generator, tokenizer, ids, max_new_tokens):
    import torch
    from exllamav3 import Job
    from exllamav3.generator.sampler.presets import ArgmaxSampler
    input_ids = torch.tensor([ids], dtype=torch.long)
    job = Job(input_ids=input_ids, max_new_tokens=max_new_tokens, sampler=ArgmaxSampler(),
              stop_conditions=[tokenizer.eos_token_id, "<|im_end|>"], decode_special_tokens=False)
    generator.enqueue(job)
    text, n = "", 0
    t0 = time.time()
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            text += r.get("text", "")
            ids_ = r.get("token_ids")
            if ids_ is not None:
                n += int(ids_.shape[-1])
    return text, n, time.time() - t0


def kl_div(p_logits, q_logits, vocab: int):
    """KL(p || q) from logits over the real vocabulary (the generator pads the
    vocab to a multiple of 32 with -inf, which turns a naive KL into NaN)."""
    import torch
    p = p_logits[:vocab].double()
    q = q_logits[:vocab].double()
    lp = torch.log_softmax(p, -1)
    lq = torch.log_softmax(q, -1)
    kl = (lp.exp() * (lp - lq))
    kl = torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
    return float(kl.sum())


def load_points(model_dir: str, rows: int, points: int, max_len: int):
    """Teacher-forced cut points from the model's qbench transcripts."""
    f = Path(model_dir) / "qbench_prompts_gen.json"
    if not f.is_file():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))
    out = []
    for r in d.get("rows", [])[:rows]:
        inp, resp = r["input_ids"], r["response_ids"]
        if not resp:
            continue
        for i in range(points):
            # cut points spread through the response, never at 0 (the prompt boundary is
            # the easiest position) and never at the very end
            frac = (i + 1) / (points + 1)
            cut = max(1, int(len(resp) * frac))
            ids = (inp + resp[:cut])[-max_len:]
            out.append((ids, resp[cut] if cut < len(resp) else None))
    return out


def make_passkey_prompt(tokenizer, total_tokens: int, key: str):
    filler = ("The grass is green. The sky is blue. The sun is yellow. Here we go. "
              "There and back again. ")
    unit = tokenizer.encode(filler, add_bos=False)[0].tolist()
    reps = max(1, total_tokens // len(unit))
    half = reps // 2
    body = (filler * half) + f"\n\nThe secret passkey is {key}. Remember it.\n\n" + (filler * (reps - half))
    msgs = [{"role": "user", "content": body + "\n\nWhat is the secret passkey? Answer with the key only."}]
    ids = tokenizer.hf_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    return ids[0].tolist()


# --------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="/".join(ALL_MODES),
                    help="slash-separated list from: " + " / ".join(ALL_MODES) + "  (e.g. fp16/4/nvfp4)")
    ap.add_argument("--context", type=int, default=16384, help="cache size for the KL/speed phase")
    ap.add_argument("--rows", type=int, default=24, help="qbench transcripts to use (24 available)")
    ap.add_argument("--points", type=int, default=6, help="cut points per transcript")
    ap.add_argument("--max-len", type=int, default=3072, help="tokens fed per cut point")
    ap.add_argument("--passkey-len", type=int, default=65536, help="filler tokens for the passkey test (0 = skip)")
    ap.add_argument("--no-passkey", action="store_true")
    ap.add_argument("--quick", action="store_true", help="4 rows, 2 points, 32k passkey")
    ap.add_argument("--out", default=str(ROOT / "kv_cache_tests.json"))
    a = ap.parse_args()
    if a.quick:
        a.rows, a.points, a.passkey_len = 4, 2, 32768
    if a.no_passkey:
        a.passkey_len = 0

    cfg = load_env()
    model_dir = cfg.get("MODEL_DIR", "models/Qwen3.8-27B-EXL3-2.0bpw")
    if not os.path.isabs(model_dir):
        model_dir = str(ROOT / model_dir)
    gpu_mem = cfg.get("GPU_MEM_GB") or "14.7"

    raw_modes = a.modes.replace(";", "/").replace("+", "/")
    if "/" not in raw_modes and "," in raw_modes and raw_modes not in ALL_MODES:
        raise SystemExit("separate modes with '/', e.g. --modes fp16/8,4/nvfp4  ('8,4' is one mode)")
    modes = []
    for m in raw_modes.split("/"):
        m = m.strip()
        if not m or m in modes:
            continue
        if m not in ALL_MODES:
            raise SystemExit(f"unknown mode {m!r}; choose from {ALL_MODES}")
        modes.append(m)
    if any(m in ("fp8", "nvfp4") for m in modes):
        try:
            import exllamav3.cache.nvfp4  # noqa: F401
        except ImportError:
            raise SystemExit("fp8/nvfp4 need the KV patch: .venv\\Scripts\\python.exe tools\\patch_kv.py apply")

    import torch
    from serve_openai import _cap_process_vram  # same VRAM behaviour as the server
    _cap_process_vram(gpu_mem)
    cc = torch.cuda.get_device_capability(0)
    log(f"GPU: {torch.cuda.get_device_name(0)}  sm_{cc[0]}{cc[1]}  budget {gpu_mem} GB")
    if any(m in ("fp8", "nvfp4") for m in modes) and cc < (8, 9):
        log("! fp8/nvfp4 Triton kernels need compute capability 8.9+ (Ada/Blackwell); expect a compile error on this GPU")

    points = load_points(model_dir, a.rows, a.points, a.max_len)
    if not points:
        log("! no qbench_prompts_gen.json in the model dir - KL phase skipped")
    log(f"KL phase: {len(points)} cut points, context {a.context}; modes: {', '.join(modes)}")

    # the baseline (fp16 if requested) must run first
    baseline_name = "fp16" if "fp16" in modes else modes[0]
    modes = [baseline_name] + [m for m in modes if m != baseline_name]
    results = {"_meta": {"gpu": torch.cuda.get_device_name(0), "cc": f"sm_{cc[0]}{cc[1]}",
                         "context": a.context, "points": len(points), "passkey_len": a.passkey_len}}
    base_logits = None
    passkey = "7391-XKQ-2048"
    for mode in modes:
        log(); log(f"=== cache format: {mode}  ({BITS[mode]} bits/elem) ===")
        t0 = time.time()
        try:
            model, cache, tokenizer, generator = build(model_dir, gpu_mem, a.context, mode)
        except Exception as e:
            log(f"  LOAD FAILED: {type(e).__name__}: {str(e)[:300]}")
            results[mode] = {"error": f"load: {type(e).__name__}: {str(e)[:300]}"}
            gc.collect(); torch.cuda.empty_cache()
            continue
        alloc, res = vram_gib()
        vocab = int(getattr(model.config, "vocab_size", None) or getattr(tokenizer, "actual_vocab_size", 0)
                    or generator.padded_vocab_size)
        r = {"bits": BITS[mode], "load_s": round(time.time() - t0, 1),
             "vram_alloc_gib": round(alloc, 2), "vram_reserved_gib": round(res, 2)}
        log(f"  loaded in {r['load_s']}s   VRAM {alloc:.2f} GiB alloc / {res:.2f} GiB reserved")
        try:
            # --- KL / top-1 / ref-acc
            if points:
                logits = []
                t1 = time.time()
                for i, (ids, _) in enumerate(points):
                    logits.append(next_token_logits(generator, tokenizer, ids))
                    if (i + 1) % 8 == 0:
                        log(f"  probed {i + 1}/{len(points)} cut points ({time.time() - t1:.0f}s)")
                if mode == baseline_name:
                    base_logits = logits
                    r["baseline"] = True
                kls, top1, refacc = [], [], []
                for (ids, ref_tok), lg, bl in zip(points, logits, base_logits or logits):
                    kls.append(kl_div(bl, lg, vocab))
                    top1.append(float(bl[:vocab].argmax() == lg[:vocab].argmax()))
                    if ref_tok is not None:
                        refacc.append(float(int(lg[:vocab].argmax()) == ref_tok))
                kls_sorted = sorted(kls)
                r.update({
                    "kl_mean": sum(kls) / len(kls),
                    "kl_p95": kls_sorted[min(len(kls) - 1, int(0.95 * len(kls)))],
                    "kl_max": kls_sorted[-1],
                    "top1_agree": sum(top1) / len(top1),
                    "ref_acc": (sum(refacc) / len(refacc)) if refacc else None,
                })
                log(f"  KL vs {baseline_name}: mean {r['kl_mean']:.5f}  "
                    f"p95 {r['kl_p95']:.5f}  max {r['kl_max']:.4f}   top-1 agree {r['top1_agree']*100:.1f}%"
                    + (f"   ref-acc {r['ref_acc']*100:.1f}%" if r["ref_acc"] is not None else ""))
            # --- decode speed
            ids2 = (points[0][0] if points else tokenizer.encode("Hello " * 800, add_bos=False)[0].tolist())[:2048]
            _txt, n, dt = generate(generator, tokenizer, ids2, 96)
            r["decode_tps"] = round(n / dt, 1) if dt > 0 else None
            log(f"  decode: {n} tokens in {dt:.1f}s = {r['decode_tps']} tok/s")
        except Exception as e:
            log(f"  PROBE FAILED: {type(e).__name__}: {str(e)[:300]}")
            r["error"] = f"probe: {type(e).__name__}: {str(e)[:300]}"
        finally:
            unload(model, cache, generator)

        # --- passkey at long context (separate load, big cache)
        if a.passkey_len and mode != "fp16" and "error" not in r:
            cs = ((a.passkey_len + 4096 + 255) // 256) * 256
            log(f"  passkey: reloading with cache {cs} ...")
            try:
                model, cache, tokenizer, generator = build(model_dir, gpu_mem, cs, mode)
                alloc, res = vram_gib()
                r["passkey_vram_alloc_gib"] = round(alloc, 2)
                ids = make_passkey_prompt(tokenizer, a.passkey_len, passkey)
                t2 = time.time()
                txt, n, dt = generate(generator, tokenizer, ids, 24)
                ok = passkey.split("-")[0] in txt and "XKQ" in txt
                r["passkey_len"] = len(ids)
                r["passkey_ok"] = ok
                r["passkey_answer"] = txt.strip()[:80]
                log(f"  passkey ({len(ids)} tokens, {time.time() - t2:.0f}s): {'OK' if ok else 'MISSED'}  -> {txt.strip()[:60]!r}")
            except Exception as e:
                log(f"  passkey FAILED: {type(e).__name__}: {str(e)[:300]}")
                r["passkey_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            finally:
                try:
                    unload(model, cache, generator)
                except Exception:
                    pass
        results[mode] = r
        Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ------------------------------------------------------------- summary ---
    log(); log("=" * 78)
    log(f"{'format':6s} {'bits':>4s} {'VRAM':>6s} {'KL mean':>9s} {'KL p95':>9s} {'top-1':>6s} {'ref':>5s} {'tok/s':>6s} {'passkey':>8s}")
    for mode in modes:
        r = results.get(mode, {})
        if "error" in r:
            log(f"{mode:6s} {BITS[mode]:4.1f}  {r['error'][:60]}")
            continue
        pk = "-" if "passkey_ok" not in r else ("OK" if r["passkey_ok"] else "MISS")
        log(f"{mode:6s} {BITS[mode]:4.1f} {r.get('vram_alloc_gib', 0):6.2f} "
            f"{r.get('kl_mean', float('nan')):9.5f} {r.get('kl_p95', float('nan')):9.5f} "
            f"{(r.get('top1_agree') or 0)*100:5.1f}% {(r.get('ref_acc') or 0)*100:4.0f}% "
            f"{r.get('decode_tps') or 0:6.1f} {pk:>8s}")
    log(f"\nfull results: {a.out}")
    # verdict for the 4.5-bit question
    if "4" in results and "nvfp4" in results and "kl_mean" in results["4"] and "kl_mean" in results["nvfp4"]:
        a4, an = results["4"]["kl_mean"], results["nvfp4"]["kl_mean"]
        better = "nvfp4" if an < a4 else "int 4"
        log(f"4.5-bit verdict: {better} has the lower KL ({min(a4, an):.5f} vs {max(a4, an):.5f}); "
            f"passkey int4={results['4'].get('passkey_ok')} nvfp4={results['nvfp4'].get('passkey_ok')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure what each quant actually costs, instead of deriving it.

`tools/profiles.py` sizes the first-run menu from a table where only the 2.0bpw
row was measured; the rest were derived from shard sizes minus embeddings, and
the KV figures come from a formula. This script replaces both with numbers from
this card:

  weights      GiB resident after the text model loads (cache subtracted out)
  vision       GiB the tower adds on top
  KV           KB per token, measured by loading the same model at two context
               sizes and differencing - not assumed from a table
  max context  the largest context that boots AND survives a full prefill
  peak         GiB the whole process holds at that context under load, as
               nvidia-smi sees it (which is what Task Manager and the driver
               see, and therefore what actually has to fit)
  speed        prefill and decode tokens/second at that size

Each measurement runs in a **fresh child process**: freeing an ExLlamaV3 model
inside one process leaves fragmentation behind, and a stale allocator makes the
next number a lie.

    python tools/bench_vram.py --list
    python tools/bench_vram.py --quant 3.0            # one quant, resumable
    python tools/bench_vram.py --all --delete-after   # overnight, ~110 GB down
    python tools/bench_vram.py --report               # rebuild the tables

Results append to bench_vram.json after every quant, so it can be stopped and
resumed, and each run prints a Markdown table (bench_vram.md) plus the exact
QUANTS rows to paste into tools/profiles.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_JSON = ROOT / "bench_vram.json"
OUT_MD = ROOT / "bench_vram.md"
GIB = 1024 ** 3
PROBE_A = 32768          # two context sizes, differenced for the KV rate
PROBE_B = 65536
NATIVE_CTX = 262144
SAFETY_GIB = 0.4
CHILD_TIMEOUT = 2400     # first load of a quant compiles kernels


# =============================================================== child ======
# Runs in its own process: load once, measure, print JSON, exit.

def child(spec: dict) -> dict:
    import torch
    import serve_openai

    def gib(x):
        return round(x / GIB, 3)

    def smi_process_gib():
        """What the driver attributes to this process - the number that has to
        fit on the card, and the one Task Manager shows."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20).stdout
            for line in out.splitlines():
                pid, used = [p.strip() for p in line.split(",")[:2]]
                if int(pid) == os.getpid():
                    return round(int(used) / 1024, 3)
        except Exception:                              # noqa: BLE001
            pass
        return None

    argv = ["-m", spec["model_dir"], "-gs", str(spec["budget"]),
            "-cs", str(spec["ctx"]), "-cq", spec["cache"]]
    if spec.get("mtp", True):
        argv.append("-mtp")

    result = {"ctx": spec["ctx"], "cache": spec["cache"], "ok": False}
    t0 = time.time()
    # Pin this process to the same budget the real server enforces. serve_openai
    # .main() calls this before build_model for a reason its own docstring gives:
    # "-gs" is an autosplit hint, not a limit, and ExLlama lifts the CUDA fraction
    # after autosplit. Without it the child may use the whole physical card, so a
    # context "verified" here could be one the server can never actually reach.
    serve_openai._cap_process_vram(spec["budget"])
    result["vram_cap_gib"] = float(spec["budget"])
    torch.cuda.reset_peak_memory_stats()
    generator, tokenizer, config = serve_openai.build_model(
        argv, use_draft=spec.get("mtp", True))
    result.update({
        "load_seconds": round(time.time() - t0, 1),
        "allocated_gib": gib(torch.cuda.memory_allocated(0)),
        "reserved_gib": gib(torch.cuda.memory_reserved(0)),
        "process_gib": smi_process_gib(),
    })

    if spec.get("vision"):
        before = torch.cuda.memory_allocated(0)
        serve_openai.load_vision(config, 1048576)
        loaded = serve_openai.vision["model"] is not None
        result["vision_gib"] = gib(torch.cuda.memory_allocated(0) - before) if loaded else 0.0
        result["vision_loaded"] = loaded

    if spec.get("stress"):
        from exllamav3 import Job
        from exllamav3.generator.sampler.presets import ComboSampler

        want = int(spec["stress_tokens"])
        # a real prompt, not a repeated token: repeated ids are the easy case
        # for a paged cache and would flatter the numbers
        seed_text = ("The quick brown fox jumps over the lazy dog near the "
                     "riverbank while seventeen swallows circle above. ") * 64
        ids = tokenizer.encode(seed_text)
        while ids.shape[-1] < want:
            ids = torch.cat([ids, ids], dim=-1)
        input_ids = ids[:, :want]
        result["prompt_tokens"] = int(input_ids.shape[-1])

        torch.cuda.reset_peak_memory_stats()
        job = Job(input_ids=input_ids, max_new_tokens=spec.get("decode_tokens", 128),
                  sampler=ComboSampler(temperature=0.6, top_k=20, top_p=0.95),
                  stop_conditions=[tokenizer.eos_token_id])
        generator.enqueue(job)
        start = time.time()
        first = None
        decoded = 0
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("stage") == "prefill":
                    continue
                # count real tokens, not text chunks - a chunk can carry several
                new = serve_openai._result_new_tokens(r)
                if new:
                    if first is None:
                        first = time.time()
                    decoded += new
        done = time.time()
        prefill_s = (first or done) - start
        decode_s = max(1e-6, done - (first or done))
        result.update({
            "prefill_seconds": round(prefill_s, 2),
            "prefill_tok_s": round(result["prompt_tokens"] / max(prefill_s, 1e-6), 1),
            "decode_tokens": decoded,
            "decode_tok_s": round(decoded / decode_s, 1),
            "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(0)),
            "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(0)),
            "peak_process_gib": smi_process_gib(),
        })

    result["ok"] = True
    return result


def run_child(spec, timeout=CHILD_TIMEOUT, verbose=True):
    """One measurement, in a process of its own."""
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child"]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=None if verbose else subprocess.DEVNULL,
                            text=True)
    try:
        out, _ = proc.communicate(json.dumps(spec), timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"ok": False, "error": f"timed out after {timeout}s"}
    for line in reversed((out or "").splitlines()):
        if line.startswith("{") and '"ok"' in line:
            try:
                got = json.loads(line)
            except ValueError:
                continue
            # The child reports its own exception as {"ok": false, "error": ...}
            # and returns here, so the text-scanning fallback below never saw it
            # and attempts[].oom was always None. Classify it here instead.
            if not got.get("ok") and "oom" not in got:
                text = str(got.get("error", "")).lower()
                got["oom"] = "out of memory" in text or "cuda error" in text
            return got
    tail = (out or "").strip().splitlines()[-3:]
    lowered = " ".join(tail).lower()
    oom = "out of memory" in lowered or "cuda error" in lowered
    return {"ok": False, "oom": oom, "error": " / ".join(tail) or "no result"}


# ============================================================== parent ======

def env_token():
    """HF_TOKEN from .env, so a gated repo does not need it typed again."""
    path = ROOT / ".env"
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("HF_TOKEN") and "=" in line:
            value = line.split("=", 1)[1].split("#")[0].strip().strip('"').strip("'")
            if value:
                return value
    return None


def load_results():
    if OUT_JSON.is_file():
        try:
            return json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {"gpu": None, "quants": {}}


def save_results(data):
    OUT_JSON.write_text(json.dumps(data, indent=1), encoding="utf-8")


def free_disk_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except OSError:
        return 0.0


def ensure_weights(quant, token=None) -> Path:
    target = ROOT / quant.model_dir
    weights = list(target.glob("*.safetensors")) if target.is_dir() else []
    if weights:
        return target
    need = quant.disk_gb + 5
    free = free_disk_gb(ROOT)
    if free < need:
        raise RuntimeError(f"{free:.0f} GB free, {quant.id}bpw needs about "
                           f"{need:.0f} GB - free some space or use --delete-after")
    print(f"  downloading {quant.repo}"
          + (f" @ {quant.revision}" if quant.revision else "")
          + f"  (~{quant.disk_gb} GB)", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=quant.repo, revision=quant.revision or None,
                      local_dir=str(target), token=token or os.environ.get("HF_TOKEN"),
                      max_workers=8)
    return target


def kv_rate_kb(probe_a: dict, probe_b: dict) -> float:
    """KB per token, from two probes of the same model at different context
    sizes. probe_a's allocated_gib is snapshotted right after build_model(),
    before the vision tower ever loads (that happens later, in its own
    before/after delta - see child()) - so it already excludes vision, and no
    vision term belongs in this difference. Subtracting one here used to
    double-count it, inflating the measured rate by vision_gib * 32 KB/token
    for the 32768-token gap between PROBE_A and PROBE_B."""
    delta = probe_b["allocated_gib"] - probe_a["allocated_gib"]
    rate = delta * GIB / (PROBE_B - PROBE_A) / 1024
    # A flat or backwards pair means the probes did not measure what they were
    # meant to (fragmentation, a partly-failed load). Clamping keeps the caller
    # from dividing by ~0, but the result is not a measurement - measure_quant
    # checks the same condition and says so in kv_source.
    return max(0.1, rate)


def kv_rate_is_suspect(probe_a: dict, probe_b: dict) -> bool:
    """True when the two probes show no real growth, so the rate is the clamp."""
    return (probe_b["allocated_gib"] - probe_a["allocated_gib"]) <= 0


def weights_from_probe(probe_a: dict, kv_kb: float) -> float:
    """Resident weights GiB, backing probe A's own KV footprint out of its
    allocated_gib. Same vision caveat as kv_rate_kb: probe_a's allocated_gib
    never included vision, so there is nothing to subtract for it here."""
    return round(probe_a["allocated_gib"] - kv_kb * PROBE_A * 1024 / GIB, 3)


def plan_context(weights_gib, kv_kb, vision_gib, budget, overhead):
    """Largest context whose KV fits in what is left, with the same 0.4 GiB of
    slack the profile planner uses."""
    free = budget - SAFETY_GIB - weights_gib - vision_gib - overhead
    if free <= 0:
        return 0
    tokens = int(free * GIB / (kv_kb * 1024))
    return min(NATIVE_CTX, max(0, tokens) // 256 * 256)


def overhead_at(peak_gib: float, weights_gib: float, vision_gib: float,
                kv_kb: float, ctx: int) -> float:
    """What the process holds beyond the tensors it was asked for, at `ctx`.

    peak_gib is the real peak during a prefill, so this captures everything the
    planner does not model: the CUDA context, kernels and workspace, the paged
    cache's page tables, CUDA graphs, and the prefill's own scratch. All of that
    was assumed to be a flat 1.7 GiB regardless of context, which is what made
    the planner offer contexts that then OOMed."""
    tensors = weights_gib + vision_gib + kv_kb * ctx * 1024 / GIB
    return round(peak_gib - tensors, 3)


def sweep_overhead(quant, budget, args, row, rel, kv_kb, weights_gib, vision_gib):
    """Measure the overhead at several context sizes, so it can be modelled as a
    function of context instead of a constant.

    One stress probe per point: the peak only shows up during a real prefill, not
    at load. Points that OOM are recorded too - the largest context that failed is
    as informative as the largest that worked."""
    points = []
    ladder = [c for c in args.sweep_points if c <= args.max_context]
    for ctx in sorted(ladder):
        stress = int(ctx * args.stress_fraction) // 256 * 256
        print(f"  [sweep] {ctx} tokens of context, prefilling {stress} ...", flush=True)
        c = run_child({"model_dir": rel, "budget": budget, "ctx": ctx,
                       "cache": args.cache, "vision": bool(vision_gib),
                       "mtp": not args.no_draft, "stress": True,
                       "stress_tokens": stress,
                       "decode_tokens": args.decode_tokens},
                      args.timeout, args.verbose)
        point = {"ctx": ctx, "ok": bool(c.get("ok"))}
        if c.get("ok"):
            peak = c.get("peak_process_gib") or c.get("peak_reserved_gib") or 0.0
            point.update({
                "peak_process_gib": c.get("peak_process_gib"),
                "peak_reserved_gib": c.get("peak_reserved_gib"),
                "peak_gib": peak,
                "overhead_gib": overhead_at(peak, weights_gib, vision_gib, kv_kb, ctx),
            })
            print(f"          peak {peak:.2f} GiB -> overhead "
                  f"{point['overhead_gib']:.2f} GiB", flush=True)
        else:
            point["error"] = c.get("error")
            print(f"          did not survive ({str(c.get('error'))[:80]})", flush=True)
        points.append(point)
    row["overhead_sweep"] = points
    return points


def fit_overhead(points) -> dict | None:
    """Least-squares fit of overhead(ctx) = base + per_token * ctx over every
    surviving point, in the units profiles.py wants: GiB of fixed cost, and KB
    per token of context-dependent cost."""
    good = [p for p in points if p.get("ok") and p.get("overhead_gib") is not None]
    if len(good) < 2:
        return None
    xs = [p["ctx"] for p in good]
    ys = [p["overhead_gib"] for p in good]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom) if denom else 0.0
    base = my - slope * mx
    return {"base_gib": round(base, 3),
            "per_token_kb": round(slope * GIB / 1024, 4),
            "points": n,
            "max_ok_ctx": max(xs),
            "max_failed_ctx": max([p["ctx"] for p in points if not p.get("ok")], default=None)}


def measure_quant(quant, budget, args, results):
    print(f"\n=== {quant.id}bpw  ({quant.note}) ".ljust(64, "=") + "\n", flush=True)
    row = {"id": quant.id, "bpw": quant.bpw, "repo": quant.repo,
           "table_gpu_gib": quant.gpu_gib, "table_vision_gib": quant.vision_gib,
           "when": time.strftime("%Y-%m-%d %H:%M")}
    try:
        model_dir = ensure_weights(quant, args.hf_token or env_token())
    except Exception as e:                             # noqa: BLE001
        row["error"] = f"download failed: {e}"
        return row
    rel = str(Path(quant.model_dir).as_posix())

    # --- probe A: the smaller context, plus the vision tower ---------------
    print(f"  [1/3] loading at {PROBE_A} tokens ...", flush=True)
    a = run_child({"model_dir": rel, "budget": budget, "ctx": PROBE_A,
                   "cache": args.cache, "vision": True, "mtp": not args.no_draft},
                  args.timeout, args.verbose)
    row["probe_a"] = a
    if not a.get("ok"):
        row["error"] = f"could not load at {PROBE_A}: {a.get('error')}"
        print(f"  FAILED: {row['error']}", flush=True)
        return row

    # --- probe B: the same model, more context -> KV per token -------------
    if args.quick:
        kv_kb = float(args.assume_kv)
        row["kv_source"] = f"assumed {kv_kb} KB/token (--quick)"
    else:
        print(f"  [2/3] loading at {PROBE_B} tokens (for the KV rate) ...", flush=True)
        b = run_child({"model_dir": rel, "budget": budget, "ctx": PROBE_B,
                       "cache": args.cache, "vision": False,
                       "mtp": not args.no_draft}, args.timeout, args.verbose)
        row["probe_b"] = b
        if not b.get("ok"):
            kv_kb = float(args.assume_kv)
            row["kv_source"] = f"probe B failed, assumed {kv_kb} KB/token"
        elif kv_rate_is_suspect(a, b):
            # more context did not cost more memory: something is wrong with the
            # probes, so fall back rather than recording 0.1 KB/token as fact
            kv_kb = float(args.assume_kv)
            row["kv_source"] = (f"probes showed no growth "
                                f"({a['allocated_gib']} -> {b['allocated_gib']} GiB), "
                                f"assumed {kv_kb} KB/token")
        else:
            kv_kb = kv_rate_kb(a, b)
            row["kv_source"] = "measured"
    row["kv_kb_per_token"] = round(kv_kb, 2)

    vision_gib = a.get("vision_gib") or 0.0
    weights_gib = weights_from_probe(a, kv_kb)
    row["weights_gib"] = weights_gib
    row["vision_gib"] = round(vision_gib, 3)
    # what the process holds beyond the tensors: CUDA context, kernels, torch
    row["overhead_gib"] = round((a.get("process_gib") or a["reserved_gib"])
                                - a["allocated_gib"], 3)

    # --- probe C: the biggest context that survives a real prefill ---------
    ctx = plan_context(weights_gib, kv_kb, vision_gib, budget,
                       max(row["overhead_gib"], 0.8))
    ctx = min(ctx, args.max_context)
    attempts = []
    for factor in (1.0, 0.85, 0.7):
        try_ctx = int(ctx * factor) // 256 * 256
        if try_ctx < 8192:
            break
        stress = int(try_ctx * args.stress_fraction) // 256 * 256
        print(f"  [3/3] {try_ctx} tokens of context, prefilling {stress} ...",
              flush=True)
        c = run_child({"model_dir": rel, "budget": budget, "ctx": try_ctx,
                       "cache": args.cache, "vision": True,
                       "mtp": not args.no_draft, "stress": True,
                       "stress_tokens": stress, "decode_tokens": args.decode_tokens},
                      args.timeout, args.verbose)
        attempts.append({"ctx": try_ctx, "ok": c.get("ok"),
                         "error": c.get("error"), "oom": c.get("oom")})
        if c.get("ok"):
            row["stress"] = c
            row["max_context_verified"] = try_ctx
            row["peak_process_gib"] = c.get("peak_process_gib") or c.get("peak_reserved_gib")
            row["headroom_gib"] = round(
                budget - (row["peak_process_gib"] or 0), 2)
            break
        print(f"        did not survive ({c.get('error')}) - stepping down",
              flush=True)
    row["attempts"] = attempts
    if "max_context_verified" not in row:
        row["error"] = "no context survived the stress prefill"

    # --- optional: how the overhead grows with context --------------------
    if args.sweep:
        points = sweep_overhead(quant, budget, args, row, rel,
                                kv_kb, weights_gib, vision_gib)
        fit = fit_overhead(points)
        if fit:
            row["overhead_fit"] = fit
            print(f"  overhead ~ {fit['base_gib']:.2f} GiB + "
                  f"{fit['per_token_kb']:.3f} KB/token  ({fit['points']} points)",
                  flush=True)
    return row


# ============================================================== report ======

def report(data):
    rows = [r for r in data["quants"].values() if not r.get("error")]
    rows.sort(key=lambda r: r["bpw"], reverse=True)
    gpu = data.get("gpu") or {}
    lines = [
        "# Measured VRAM and context per quant", "",
        f"Card: **{gpu.get('name', '?')}**, {gpu.get('memory', '?')} GB, "
        f"driver {gpu.get('driver', '?')}, budget {gpu.get('budget', '?')} GiB, "
        f"KV cache `{gpu.get('cache', '?')}`, MTP {gpu.get('draft', 'on')}.",
        f"Generated by `tools/bench_vram.py` on {time.strftime('%Y-%m-%d')}.", "",
        "| bpw | weights GiB | vision GiB | KV KB/tok | max context | peak process GiB "
        "| headroom GiB | prefill tok/s | decode tok/s | table said |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        s = r.get("stress") or {}
        lines.append(
            f"| {r['id']} | {r.get('weights_gib', '?')} | {r.get('vision_gib', '?')} "
            f"| {r.get('kv_kb_per_token', '?')} | {r.get('max_context_verified', '?')} "
            f"| {r.get('peak_process_gib', '?')} | {r.get('headroom_gib', '?')} "
            f"| {s.get('prefill_tok_s', '?')} | {s.get('decode_tok_s', '?')} "
            f"| {r.get('table_gpu_gib', '?')} |")
    failed = [r for r in data["quants"].values() if r.get("error")]
    if failed:
        lines += ["", "## Did not complete", ""]
        for r in failed:
            lines.append(f"- **{r['id']}bpw**: {r['error']}")

    lines += ["", "## Rows for tools/profiles.py", "",
              "Measured `gpu_gib` and `vision_gib`; everything else unchanged:", "",
              "```python"]
    for r in rows:
        lines.append(
            f'    Quant("{r["id"]}", {r["bpw"]}, "{r["repo"]}", ..., '
            f'{r.get("weights_gib")}, ..., {r.get("vision_gib")}, ...),')
    lines += ["```", ""]
    if rows:
        kv = [r["kv_kb_per_token"] for r in rows if r.get("kv_kb_per_token")]
        import profiles as _p
        base_lo, base_hi = min(kv) / _p.MTP_FACTOR, max(kv) / _p.MTP_FACTOR
        lines.append(
            f"Measured KV: {min(kv)}-{max(kv)} KB/token with MTP loaded, i.e. a base "
            f"rate of {base_lo:.2f}-{base_hi:.2f} KB/token once the {_p.MTP_FACTOR:.4f} "
            f"MTP factor is divided out (profiles.py's KV_KB assumes "
            f"{_p.KV_KB['4']} for cache `4` and applies that factor separately).")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"\n  full table -> {OUT_MD}")


# ================================================================ main ======

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--quant", action="append", default=[],
                    help="bpw id to measure, e.g. 3.0 (repeatable)")
    ap.add_argument("--all", action="store_true", help="every quant in the table")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true", help="rebuild the tables only")
    ap.add_argument("--redo", action="store_true", help="re-measure quants already done")
    ap.add_argument("--delete-after", action="store_true",
                    help="remove the weights once a quant is measured")
    ap.add_argument("--quick", action="store_true",
                    help="skip the second load; assume the KV rate")
    ap.add_argument("--assume-kv", type=float, default=round(18.0 * 17 / 16, 2),
                    help="KB per token when not measured. profiles.py's KV_KB is the "
                         "base rate and it applies the 17/16 MTP factor separately; the "
                         "bench measures with MTP loaded, so its rate already includes "
                         "it - hence 19.13, not 18 (default: 19.13, cache 4)")
    ap.add_argument("--cache", default="4", help="KV cache format (default 4)")
    ap.add_argument("--no-draft", action="store_true", help="measure without MTP")
    ap.add_argument("--stress-fraction", type=float, default=0.9,
                    help="how much of the context to prefill (default 0.9)")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--max-context", type=int, default=NATIVE_CTX)
    ap.add_argument("--budget", type=float, default=None,
                    help="VRAM budget in GiB (default: profiles.py's rule)")
    ap.add_argument("--timeout", type=int, default=CHILD_TIMEOUT)
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--sweep", action="store_true",
                    help="also measure the overhead at several context sizes, so it "
                         "can be modelled as a function of context instead of a constant")
    ap.add_argument("--sweep-points", type=lambda v: [int(x) for x in v.split(",")],
                    default=[32768, 65536, 131072, 196608, 262144],
                    help="context sizes for --sweep (comma separated)")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the plumbing with fabricated measurements")
    args = ap.parse_args(argv)

    if args.dry_run:
        # A dry run fabricates every number it prints. It must never share a file
        # with the real results, which can hold hours of overnight measurements
        # and which --all resumes from: it goes through the same load/save/report
        # path, so pointing it at the real file merged invented rows into it and
        # overwrote the recorded GPU block.
        global OUT_JSON, OUT_MD
        OUT_JSON = ROOT / "bench_vram.dryrun.json"
        OUT_MD = ROOT / "bench_vram.dryrun.md"

    if args.child:
        spec = json.loads(sys.stdin.read())
        try:
            print(json.dumps(child(spec)), flush=True)
        except Exception as e:                         # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}),
                  flush=True)
        return 0

    import profiles
    data = load_results()
    if args.report:
        report(data)
        return 0

    quants = {q.id: q for q in profiles.QUANTS}
    if args.list:
        print(f"{'bpw':>5}  {'download':>9}  {'table GiB':>9}  measured")
        for q in profiles.QUANTS:
            done = data["quants"].get(q.id, {})
            state = (done.get("error") and "failed") or done.get("weights_gib") or "-"
            print(f"{q.id:>5}  {q.disk_gb:>7} GB  {q.gpu_gib:>9}  {state}")
        return 0

    # ascending bpw: 2.0 is already downloaded, so the harness proves itself
    # before anything spends 20 GB of bandwidth
    ordered = [q.id for q in sorted(profiles.QUANTS, key=lambda q: q.bpw)]
    wanted = ordered if args.all else [q for q in args.quant if q in quants]
    if not wanted:
        ap.error("say which: --quant 3.0 (repeatable), or --all, or --list")

    if args.dry_run:
        gpu = type("G", (), {"name": "Dry run", "total_gib": 32.0, "driver": "0"})()
    else:
        gpu = profiles.detect_gpu(profiles.read_env(profiles.ENV_FILE))
    budget = args.budget or profiles.budget_gib(float(gpu.total_gib))
    data["gpu"] = {"name": gpu.name, "memory": float(gpu.total_gib),
                   "driver": getattr(gpu, "driver", "?"), "budget": budget,
                   "cache": args.cache, "draft": "off" if args.no_draft else "mtp"}
    print(f"card {gpu.name}  {gpu.total_gib} GB   budget {budget} GiB   "
          f"cache {args.cache}   {'no draft' if args.no_draft else 'MTP'}")

    for qid in wanted:
        if qid in data["quants"] and not args.redo and not data["quants"][qid].get("error"):
            print(f"\n=== {qid}bpw already measured (use --redo) ===")
            continue
        if args.dry_run:
            row = {"id": qid, "bpw": quants[qid].bpw, "repo": quants[qid].repo,
                   "weights_gib": quants[qid].gpu_gib, "vision_gib": 0.9,
                   "kv_kb_per_token": 18.4, "max_context_verified": 131072,
                   "peak_process_gib": 20.1, "headroom_gib": 9.4,
                   "table_gpu_gib": quants[qid].gpu_gib,
                   "stress": {"prefill_tok_s": 2100.0, "decode_tok_s": 71.2}}
        else:
            row = measure_quant(quants[qid], budget, args, data)
        data["quants"][qid] = row
        save_results(data)
        if row.get("error"):
            print(f"  {qid}bpw: FAILED - {row['error']}", flush=True)
        else:
            stress = row.get("stress") or {}
            print(f"  {qid}bpw: weights {row.get('weights_gib')} GiB, "
                  f"KV {row.get('kv_kb_per_token')} KB/token, "
                  f"max context {row.get('max_context_verified')}, "
                  f"peak {row.get('peak_process_gib')} GiB, "
                  f"headroom {row.get('headroom_gib')} GiB, "
                  f"{stress.get('prefill_tok_s')} prefill / "
                  f"{stress.get('decode_tok_s')} decode tok/s", flush=True)
        if args.delete_after and not row.get("error"):
            folder = ROOT / quants[qid].model_dir
            if folder.is_dir() and qid != "2.0":       # keep the shipped baseline
                shutil.rmtree(folder, ignore_errors=True)
                print(f"  removed {folder}")
        report(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())

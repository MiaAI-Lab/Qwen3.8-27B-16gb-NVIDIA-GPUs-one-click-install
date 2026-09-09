#!/usr/bin/env python3
"""GPU-aware profile picker for the Qwen3.8-27B EXL3 kit.

Called by windows\\start.bat (via win_start.py) and linux/start.sh before anything is
downloaded. It reads the GPU's VRAM with nvidia-smi, works out which quant /
context / KV-cache combinations fit under the card's budget, shows two or three
choices (best quality vs. longest context) and writes the choice into .env:

    PROFILE, MODEL_DIR, HF_TARGET_REPO, HF_REVISION, MODEL_ID,
    CONTEXT_SIZE, CACHE_QUANT, GPU_MEM_GB, VISION

Runs when .env has no PROFILE line (fresh install) or PROFILE=ask, or when the
launcher is started with `profile` / `--profile`. Later starts keep the choice.
No third-party dependencies (system Python is enough).

    python tools/profiles.py                # interactive (what the launchers do)
    python tools/profiles.py --vram 24      # pretend a 24 GB card
    python tools/profiles.py --list --vram 16
    python tools/profiles.py --auto         # take the recommended option, no prompt

Memory model (all GiB; see the README's profile table for the measurements):
    need = weights(bpw) + kv_per_token(cache) * context * 17/16   (MTP draft cache = 1/16)
         + vision tower (0.87 measured; 0.17 for the 3-bit SC quant)    if VISION=auto
         + 2.6 overhead (CUDA workspace, GDN state, allocator reserve)
    must be <= GPU_MEM_GB - 0.4 safety,  GPU_MEM_GB = total VRAM - max(1.3 GiB, 8 %)

Where the context was grown until a real prefill failed under
a stated budget, the row carries that ceiling (verified_ctx / verified_ctx_vision /
verified_budget) and a card that size is offered the measurement instead - see
max_ctx().
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


def on_disk(model_dir: str | Path) -> dict:
    """What is in a model folder, as tools/downloader.py sees it.

    Everything that asks "have we got this one?" comes through here, because
    the cheap version of the question - does config.json exist - is true of a
    download that stopped after its first few small files, and that is exactly
    the download a user needs to be told about.
    """
    import downloader
    if not model_dir:
        return {"state": "missing", "reason": "no model folder set", "bytes": 0,
                "missing": []}
    p = Path(model_dir)
    if not p.is_absolute():
        p = ROOT / p
    return downloader.folder_state(p)

# ---------------------------------------------------------------- quants -----
# gpu_gib: weights resident on the GPU (embeddings stay in system RAM).
# 2.0 SC measured (6.4); the others derived from turboderp's shard sizes minus the
# bf16 embeddings and vision tower. kl = mean KL vs the bf16 model (turboderp's
# chart; 2.0 SC is the self-calibrated variant and is probably a little better).
from collections import namedtuple
# measured:     gpu_gib / vision_gib came from a real load on real hardware
#               (measured at load), not from shard sizes.
#               A row with measured=False is shown in the menu marked as an estimate.
# verified_ctx: the largest context that survived a real full-context prefill on the
#               bench card, images off. It is a known-good floor, not a hard ceiling of
#               the quant - a rerun on an idle GPU can raise it. None = never verified,
#               so only the formula below decides.
# verified_ctx_vision: the same measurement with the vision tower loaded, which is the
#               smaller number (the tower costs ~0.8 GiB at peak). None = never measured
#               separately, and then verified_ctx answers for both - the rows measured
#               before the two were split were run with images on, so their
#               single figure is already the conservative one.
# verified_budget: the VRAM budget those ceilings were measured under. It is what turns
#               a measurement into a plan: on a budget at least this large the context is
#               a fact about the card rather than a cap borrowed from a bigger one, so
#               max_ctx() offers it instead of the formula's more cautious answer.
#               None = the number is only ever used to lower the formula.
# verified_peak_gib: the highest VRAM the bench actually saw at those ceilings. It is
#               what the row costs when the plan is the measurement, because the flat
#               2.6 GiB overhead is a bound over every quant and at a measured ceiling
#               it prices the row above the budget it was measured under.
# never_exceeded: no run at any budget has ever prefilled more than verified_ctx for
#               this quant, so it caps the plan on every card, not just at the budget
#               it was measured under. True for 5.0 and 6.0: the 29.3 GiB bench found
#               nothing above 183296 for 5.0 and nothing at all for 6.0 (contended,
#               with ~40 other PIDs on the card), and the clean 22.1 GiB run reached
#               204800 / 83712. Rows without it have been measured past their figure
#               on a roomier card, so above their budget the formula plans.
Quant = namedtuple(
    "Quant",
    "id bpw repo revision model_dir gpu_gib disk_gb vision_gib kl note measured "
    "verified_ctx verified_ctx_vision verified_budget verified_peak_gib never_exceeded",
    defaults=(False, None, None, None, None, False))
QUANTS = [
    # 6.0 / 5.0 / 4.0 measured at 22.1 GiB - a 24 GB card - on 2026-09-06. 6.0bpw had
    # never been verified anywhere: the 29.3 GiB bench found no context that survived,
    # and a clean run at a *tighter* budget prefilled 83712. 5.0bpw's old 183296 fell
    # the same way - 22.1 GiB held 204800 - so both of those numbers were contention,
    # not ceilings. What the two rows keep is never_exceeded: nothing has been past
    # 204800 / 83712 at any budget, so they do not plan past it until a 29.3 GiB run
    # says otherwise.
    Quant("6.0", 6.0, "turboderp/Qwen3.8-27B-exl3", "6.00bpw", "models/Qwen3.8-27B-EXL3-6.0bpw", 18.935, 22.9, 0.874, 0.007, "near-lossless", True, 83712, 57088, 22.1, 22.084, True),
    Quant("5.0", 5.0, "turboderp/Qwen3.8-27B-exl3", "5.00bpw", "models/Qwen3.8-27B-EXL3-5.0bpw", 16.123, 19.9, 0.877, 0.014, "excellent", True, 204800, 179712, 22.1, 22.037, True),
    Quant("4.0", 4.0, "turboderp/Qwen3.8-27B-exl3", "4.00bpw", "models/Qwen3.8-27B-EXL3-4.0bpw", 13.283, 16.9, 0.873, 0.052, "very good", True, 262144, 262144, 22.1, 21.289),
    # 3.5 / 3.0 reached the full 262144 on the 29.3 GiB bench (2026-09-03) - a cap at
    # NATIVE_CTX, which is no cap at all - so the numbers worth carrying are the ones
    # Measured under 14.7 GiB (2026-09-06). 3.5bpw is the sharpest case
    # in the table: the formula prices it out of a 16 GB card entirely (it plans 0
    # tokens) and the bench watched it prefill 77824 at a peak of 14.578.
    Quant("3.5", 3.5, "turboderp/Qwen3.8-27B-exl3", "3.50bpw", "models/Qwen3.8-27B-EXL3-3.5bpw", 11.864, 15.3, 0.872, 0.08,  "very good", True, 77824, 41984, 14.7, 14.578),
    Quant("3.0", 3.0, "turboderp/Qwen3.8-27B-exl3", "3.00bpw", "models/Qwen3.8-27B-EXL3-3.0bpw", 10.422, 13.8, 0.870, 0.112, "better", True, 148480, 117760, 14.7, 14.695),
    # 2.5bpw: weights are the fitted intercept of allocated_gib over seven contexts
    # (2026-09-06, measured) - 9.001, not the 8.3 the shard sizes suggested.
    # The tower never shows up in allocated_gib (it loads lazily), but the images-on
    # peak at 212224 sits 0.814 GiB above the text-only peak at the same context, so
    # the 0.9 estimate stands and is on the safe side.
    Quant("2.5", 2.5, "turboderp/Qwen3.8-27B-exl3", "2.50bpw", "models/Qwen3.8-27B-EXL3-2.5bpw",  9.001, 12.3, 0.9, 0.299, "good", True, 212224, 176128, 14.7, 14.688),
    Quant("2.0", 2.0, "Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw", "", "models/Qwen3.8-27B-EXL3-2.0bpw",  7.082, 9.7, 0.173, 0.35,  "fair (16 GB baseline)", True, 262144),
]
# ------------------------------------------------------------------ cards ----
# Real cards, for the setup page's simulation mode: "what would this kit offer
# on a 16 GB 5070 Ti?" without owning one. vram is the advertised size; cc is
# the CUDA compute capability, which is what decides support (see gpu_support).
# Compute capabilities from NVIDIA's own table (developer.nvidia.com/cuda/gpus):
# Blackwell 12.0, Ada 8.9, Ampere 8.6 (GA10x), Turing 7.5, Pascal 6.1.
Card = namedtuple("Card", "name vram_gib cc gen")
CARDS = [
    # Blackwell - every consumer card here is cc 12.0
    Card("RTX 5090", 32, 12.0, "Blackwell"),
    Card("RTX 5080", 16, 12.0, "Blackwell"),
    Card("RTX 5070 Ti", 16, 12.0, "Blackwell"),
    Card("RTX 5070", 12, 12.0, "Blackwell"),
    Card("RTX 5060 Ti 16GB", 16, 12.0, "Blackwell"),
    Card("RTX 5060 Ti 8GB", 8, 12.0, "Blackwell"),
    # Ada
    Card("RTX 4090", 24, 8.9, "Ada"),
    Card("RTX 4080", 16, 8.9, "Ada"),
    Card("RTX 4070 Ti SUPER", 16, 8.9, "Ada"),
    Card("RTX 4060 Ti 16GB", 16, 8.9, "Ada"),
    Card("RTX 4060 Ti 8GB", 8, 8.9, "Ada"),
    # Ampere
    Card("RTX 3090", 24, 8.6, "Ampere"),
    Card("RTX 3080 10GB", 10, 8.6, "Ampere"),
    Card("RTX 3060 12GB", 12, 8.6, "Ampere"),
    # Turing - runs, but roughly half the speed per GB/s
    Card("RTX 2080 Ti", 11, 7.5, "Turing"),
    # Pascal - below the floor, kept so the page can show what "no" looks like
    Card("GTX 1080 Ti", 11, 6.1, "Pascal"),
]

# Which KV cache formats a card can use. The integer lanes are plain bit-packed
# int32 plus Triton kernels - there is no compute-capability gate on them
# anywhere in exllamav3/cache/, so they run on every card this kit supports.
KV_KB = {"8": 34, "8,4": 26, "4": 18}       # KB per token, 16 attention layers, incl. scales
NATIVE_CTX = 262144
# Everything the process holds that is not weights, KV or the vision tower: the
# CUDA context, kernels and workspace, the paged cache's page tables, CUDA graphs
# and the prefill's own scratch. Measured against real peaks (
# peak_process_gib during a full prefill):
#
#     bpw   context   predicted   actual peak   under by
#     4.0    262144      20.64        21.29       0.65
#     3.5    262144      19.22        19.95       0.73
#     3.0    262144      17.77        18.09       0.32
#     5.0    183296      22.04        22.51       0.47
#     2.0    262144      13.74        13.75       0.01
#
# At 1.7 the planner under-predicted every row by more than MARGIN_GIB covers,
# which is how a profile that "fits" runs out of VRAM. 2.6 bounds all of them.
# The per-token term is the part that grows with context; these five points are
# each a single context per quant, so they cannot separate it from quant-to-quant
# variation - a sweep of one quant at several contexts,
# which is what can. Until it has run, the flat term carries the whole cost.
OVERHEAD_GIB = 2.6
OVERHEAD_KB_PER_TOKEN = 0.0
MARGIN_GIB = 0.4          # never plan closer than this to the budget
MIN_CTX = 32768           # below this a profile is not worth offering
# What counts as "enough context to stop trading quality for it" when picking the
# default. 131072 tokens is roughly a 400-page document; past that almost nobody in
# a chat is limited by context, and below it context is what people actually run out
# of. It was lowered to 65536 on 2026-09-02 for one reason: on a 16 GB card only the
# two *worst* quants could reach 128k, so the threshold recommended 2.0bpw (KL 0.35)
# over 3.0bpw (KL 0.112) - three times the error, for context nobody would reach.
# Measuring 2.5bpw removed that reason. It holds 176128 tokens with images on 14.7
# GiB (bench, 2026-09-06), so the middle rung clears the bar and the 16 GB default is
# 2.5bpw @ 176k with images instead of 3.0bpw @ 66k text-only - 66k cleared 65536 by
# the width of a rounding, and paid for it with images and two thirds of the context.
COMFORTABLE_CTX = 131072
MTP_FACTOR = 17 / 16


def budget_gib(total_gib: float) -> float:
    return round(total_gib - max(1.3, 0.08 * total_gib), 1)


def kv_gib(ctx: int, cache: str) -> float:
    return ctx * KV_KB[cache] * 1024 / 1024 ** 3 * MTP_FACTOR


def overhead_gib(ctx: int) -> float:
    """Non-tensor VRAM at this context: a fixed cost plus, once `--sweep` has
    measured it, a part that grows with context."""
    return OVERHEAD_GIB + OVERHEAD_KB_PER_TOKEN * ctx * 1024 / 1024 ** 3


def need_gib(q, ctx: int, cache: str, vision: bool) -> float:
    return (q.gpu_gib + kv_gib(ctx, cache)
            + (q.vision_gib if vision else 0.0) + overhead_gib(ctx))


def verified_ctx_for(q, vision: bool) -> int | None:
    """The largest context a real prefill has survived for this quant, for the
    images setting being planned. A row measured only once was measured with the
    tower loaded, so that figure answers for both."""
    if vision:
        return q.verified_ctx_vision or q.verified_ctx
    return q.verified_ctx


def measured_ctx(q, vision: bool, budget: float) -> int | None:
    """The verified context when it is a measurement of *this* budget rather than
    a ceiling inherited from a roomier card: the context was grown
    until a real prefill failed under `verified_budget` GiB, so a budget at least
    that large is known to hold it. Below that budget the number only ever caps."""
    ctx = verified_ctx_for(q, vision)
    if ctx and q.verified_budget is not None and budget >= q.verified_budget - 1e-9:
        return ctx
    return None


def need_for(q, ctx: int, cache: str, vision: bool, budget: float) -> float:
    """What a planned profile actually costs. Normally the formula; for a context
    the bench measured, the peak the bench saw. Without this the menu prices its
    own measurement out of the card - 2.5bpw at its watched 176128 tokens reads
    15.7 GB against the 14.7 GB budget it was measured under."""
    proven = measured_ctx(q, vision, budget)
    if proven is not None and ctx == proven and q.verified_peak_gib:
        return q.verified_peak_gib
    return need_gib(q, ctx, cache, vision)


def max_ctx(q, cache: str, vision: bool, budget: float) -> int:
    proven = measured_ctx(q, vision, budget)
    free = budget - MARGIN_GIB - q.gpu_gib - (q.vision_gib if vision else 0.0) - OVERHEAD_GIB
    if free <= 0:
        # The formula says the weights plus the flat overhead already fill the card.
        # That is exactly the claim a real prefill can refute, and for 3.5bpw on 14.7
        # GiB it did: 0 tokens planned, 77824 measured at a 14.578 GiB peak. Where the
        # bench has spoken for this budget it decides; otherwise the row is dropped.
        return proven or 0
    # both the KV cache and the context-dependent part of the overhead grow per
    # token, so they divide the same free space - keep this the exact inverse of
    # need_gib(), or the planner offers a context it has just priced as too big
    per_token_kb = KV_KB[cache] * MTP_FACTOR + OVERHEAD_KB_PER_TOKEN
    ctx = int(free * 1024 ** 3 / (per_token_kb * 1024))
    ctx = min(NATIVE_CTX, ctx // 256 * 256)
    # Never offer a context that a real prefill has already failed at. The formula
    # counts weights + KV + a flat overhead, but the prefill workspace and CUDA
    # graphs grow with context too - which is why 5.0bpw computes a full 262144
    # and only survived 183296 on the bench card.
    if proven:
        # The ceiling was measured under a budget no larger than this one, so this
        # card holds at least that much and the measurement is a *floor* here. The
        # flat overhead is deliberately generous (2.6 GiB bounds the worst row), so
        # on the quant that was measured it costs real context: 2.5bpw plans 186368
        # tokens on 14.7 GiB and was watched holding 212224. A number a prefill
        # survived beats a number a formula rounded up. On a bigger card the formula
        # takes over again - a ceiling reached by running out of a 14.7 GiB budget
        # says nothing about 22 GiB, and pinning it there would have offered 2.5bpw
        # less context than 3.0bpw above it.
        ctx = max(ctx, proven)
    seen = verified_ctx_for(q, vision)
    # ...unless nothing has ever gone past that number. 5.0bpw and 6.0bpw are the two
    # rows where every larger context that was tried failed, so the formula's answer
    # above their budget is not a plan, it is a guess at the thing that already broke.
    if seen and (q.never_exceeded or q.verified_budget is None):
        ctx = min(ctx, seen)
    return min(NATIVE_CTX, ctx)


def nice_ctx(ctx: int) -> int:
    """Round a context size down to a tidy multiple (keeps a little slack)."""
    if ctx >= NATIVE_CTX:
        return NATIVE_CTX
    for step in (16384, 8192, 4096, 2048):
        if ctx >= step * 2:
            return ctx // step * step
    return ctx // 256 * 256


# --------------------------------------------------------------- planner -----

def plan_one(q, cache: str, budget: float, want_vision: bool | None = None,
             floor: int = MIN_CTX) -> tuple[int, bool] | None:
    """The context-and-images decision for a single quant: (context, vision), or
    None when it cannot reach `floor` tokens under this budget.

    Every caller that plans a profile goes through here - the console menu, the
    web setup page and the web UI's model switcher - because when the rule was
    written out twice they drifted: the switcher used `vision = ctx >= 32768`
    and the menu the quarter-of-context rule below, so switching model silently
    turned images on and halved the context the picker had chosen."""
    c_off = max_ctx(q, cache, False, budget)
    c_on = max_ctx(q, cache, True, budget)

    def tidy(ctx: int, vision: bool) -> int:
        """Round down to a tidy multiple - unless this number *is* the measurement,
        in which case the rounding would hand back up to 16k tokens the card was
        watched holding. The slack nice_ctx() buys is worth having over a formula;
        over an observation it is just loss."""
        return ctx if ctx == measured_ctx(q, vision, budget) else nice_ctx(ctx)

    if want_vision is True:
        return (tidy(c_on, True), True) if c_on >= floor else None
    if want_vision is False:
        return (tidy(c_off, False), False) if c_off >= floor else None
    if c_off < floor:
        return None
    if c_on >= floor and c_on >= 0.75 * c_off:
        return tidy(c_on, True), True
    return tidy(c_off, False), False


def recommend(options: list[dict]) -> dict | None:
    """Which row a menu defaults to: the best quality that still has >= 128k of
    context, else the longest context - but only ever from rows the bench has
    stood behind. Recommending a row the kit merely computed is how a first run
    turns into an OOM: on a 32 GB card the top row is 6.0bpw, whose full-context
    prefill has never survived.

    Shared by the setup menu and the start-time picker, so "recommended" and
    "what Enter starts" cannot drift apart - which they would the moment there
    were two menus asking one question."""
    solid = [o for o in options if o.get("measured") and o.get("verified_ctx")]
    for pool in (solid, options):
        if not pool:
            continue
        # options are highest quality first, so this is "the best model that
        # still has enough context", falling back to the longest context when
        # nothing reaches it
        roomy = [o for o in pool if o["ctx"] >= COMFORTABLE_CTX]
        rec = next(iter(roomy), None) or max(pool, key=lambda o: o["ctx"])
        # ...except that the top roomy row can be compromised in two ways the row
        # directly under it is not, and both are worth exactly one rung of quality:
        #
        #   images  - it bought its context by dropping the vision tower (plan_one()
        #             does that when images cost more than a quarter of the context).
        #             Images are a capability, not a setting: a default without them
        #             means finding out by pasting a screenshot and being told no.
        #   native  - it cannot reach the model's full 262144 window and the next one
        #             can. Full context is a step change, not more of the same - no
        #             document, repo or transcript ever gets truncated - and it is
        #             what a 24 GB card gets by taking 4.0bpw @ 262k with images over
        #             5.0bpw @ 176k with them (KL 0.052 vs 0.014).
        #
        # One rung, no further: this is a step, not a slide down the table, and the
        # step never gives up images to buy context.
        if rec in roomy and roomy.index(rec) + 1 < len(roomy):
            nxt = roomy[roomy.index(rec) + 1]
            keeps_images = nxt["vision"] or not rec["vision"]
            gains_images = nxt["vision"] and not rec["vision"]
            gains_native = nxt["ctx"] >= NATIVE_CTX > rec["ctx"] and keeps_images
            if gains_images or gains_native:
                rec = nxt
        return rec
    return None


def plan(total_gib: float, want_vision: bool | None = None) -> tuple[float, list[dict]]:
    """Return (budget, options): one row per quant that fits with >= 32k context,
    highest quality first. KV cache is always the stock int4 (runs on every
    supported GPU, measured within 0.001 KL of fp16).

    want_vision decides what the images column is allowed to be:
        True  - every option carries the vision tower; a quant that cannot reach
                32k context with it loaded is dropped rather than silently
                offered without images the caller asked for.
        False - images off everywhere, so all of the VRAM goes to context.
        None  - the automatic rule (the default, and what --auto uses): keep
                images when they cost less than a quarter of the context."""
    budget = budget_gib(total_gib)
    options = []
    for q in QUANTS:
        picked = plan_one(q, "4", budget, want_vision)
        if picked is None:
            continue
        ctx, vision = picked
        options.append({
            "quant": q.id, "bpw": q.bpw, "repo": q.repo, "revision": q.revision,
            "model_dir": q.model_dir, "ctx": ctx, "cache": "4", "vision": vision,
            "need": round(need_for(q, ctx, "4", vision, budget), 1), "disk_gb": q.disk_gb, "kl": q.kl,
            "note": q.note, "budget": budget, "recommended": False,
            "measured": q.measured, "verified_ctx": verified_ctx_for(q, vision),
            "verified_budget": q.verified_budget,
        })
    rec = recommend(options)
    if rec is not None:
        rec["recommended"] = True
    return budget, options


# -------------------------------------------------------------- hardware -----

class GPU:
    def __init__(self, name="", total_gib=0.0, cc=0.0, driver=""):
        self.name, self.total_gib, self.cc, self.driver = name, total_gib, cc, driver

    @property
    def arch(self) -> str:
        c = self.cc
        if c >= 12.0: return "Blackwell"
        if c >= 10.0: return "Blackwell (datacenter)"
        if c >= 9.0:  return "Hopper"
        if c >= 8.9:  return "Ada Lovelace"
        if c >= 8.0:  return "Ampere"
        if c >= 7.5:  return "Turing"
        if c >= 7.0:  return "Volta"
        if c >= 6.0:  return "Pascal"
        return "pre-Pascal" if c else "unknown"


def detect_gpu() -> GPU:
    """GPU 0 via nvidia-smi: name, total VRAM (GiB), compute capability, driver."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            parts = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            # the name itself may contain commas: take the last three fields as numbers
            driver = parts[-1]
            cc = float(parts[-2]) if parts[-2].replace(".", "").isdigit() else 0.0
            mib = float(parts[-3])
            name = ",".join(parts[:-3]).strip()
            if mib > 0:
                return GPU(name, mib / 1024, cc, driver)
    except (FileNotFoundError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    # older drivers do not know compute_cap: fall back to name + memory only
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            name, mem = [x.strip() for x in r.stdout.strip().splitlines()[0].rsplit(",", 1)]
            return GPU(name, float(mem) / 1024, 0.0, "")
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return GPU()


def gpu_support(g: GPU) -> tuple[str, list[str]]:
    """('ok' | 'slow' | 'unsupported' | 'unknown', notes). What the engine needs:
    - CUDA 13 PyTorch wheels (the kit default) carry no code for compute capability
      < 7.5: Pascal / Volta (GTX 10xx, Titan V) cannot run it at all.
    - Turing (7.5: RTX 20xx, GTX 16xx, T4) runs but without Ampere's async copies /
      bf16: expect roughly half the speed of an Ampere+ card of the same bandwidth.
    - The integer KV cache (8 / 8,4 / 4) has no GPU-generation requirement.
    - Driver: CUDA 13 wheels need >= 580; 570-579 can use the cu128 wheels instead
      (the launcher picks the index automatically)."""
    notes = []
    if not g.name:
        return "unknown", ["no NVIDIA GPU found via nvidia-smi (this kit needs an NVIDIA card with CUDA)"]
    if g.cc and g.cc < 7.5:
        return "unsupported", [f"{g.arch} GPU (compute capability {g.cc}): CUDA 13 / current PyTorch "
                               "ship no kernels for it - this kit cannot run on this card"]
    status = "ok"
    if g.cc and g.cc < 8.0:
        status = "slow"
        notes.append(f"{g.arch} GPU (sm_{int(g.cc*10)}): supported, but about half the speed of Ampere or newer")
    if g.driver:
        try:
            major = int(g.driver.split(".")[0])
            if major < 570:
                status = "unsupported" if major < 550 else status
                notes.append(f"driver {g.driver} is too old for the CUDA 12.8/13 PyTorch wheels - update the NVIDIA driver (>= 580 recommended)")
            elif major < 580:
                notes.append(f"driver {g.driver}: CUDA 13 wheels need >= 580; the launcher falls back to the cu128 wheels")
        except ValueError:
            pass
    return status, notes


def kv_support(cc: float) -> list[dict]:
    """Which KV cache formats this card can run.

    All of them: the integer lanes have no hardware requirement at all, and the
    int4 cache - the kit's default - is also the cheapest per token."""
    return [{"format": fmt, "kb_per_token": KV_KB[fmt], "label": label,
             "available": True, "why": ""}
            for fmt, label in (("4", "int4 - the kit default"),
                               ("8,4", "int8 keys, int4 values"),
                               ("8", "int8"))]


def simulate(vram_gib: float, cc: float = 0.0, driver: str = "",
             name: str = "", want_vision=None) -> dict:
    """What this kit would offer on a card it is not running on.

    Same planner, same support rules, no side effects - nothing here reads or
    writes .env, so the setup page can explore freely mid-install."""
    g = GPU(name or f"{vram_gib:.0f} GB card", float(vram_gib), float(cc or 0.0), driver)
    status, notes = gpu_support(g)
    budget = budget_gib(g.total_gib)
    menus = {}
    for key, want in (("auto", None), ("on", True), ("off", False)):
        _, options = plan(g.total_gib, want)
        if want is True and not options:
            options = plan(g.total_gib, False)[1]      # same fallback the page uses
        rec = next((o for o in options if o["recommended"]), None)
        menus[key] = {"options": options,
                      "recommended": rec["quant"] if rec else None}
    picked = menus["auto" if want_vision is None else ("on" if want_vision else "off")]

    # What a card would have to be for the smallest profile to fit at all. Useful
    # precisely when nothing fits: "needs more" is a dead end, "needs about a
    # 13 GB card" tells someone what to look for.
    smallest = min(QUANTS, key=lambda q: q.gpu_gib)
    need = need_gib(smallest, MIN_CTX, "4", False) + MARGIN_GIB
    # invert budget_gib: total - max(1.3, 8% of total) >= need
    min_card = need + 1.3 if (need + 1.3) <= 16.25 else need / 0.92

    return {
        "gpu": {"name": g.name, "vram_gib": round(g.total_gib, 1), "cc": g.cc,
                "arch": g.arch, "driver": g.driver},
        "support": status, "notes": notes,
        "budget": round(budget, 1),
        "kv": kv_support(g.cc),
        "menus": menus,
        "options": picked["options"], "recommended": picked["recommended"],
        "vision": want_vision,
        "fits": bool(picked["options"]),
        "min_card_gib": round(min_card, 1),
        "smallest": {"quant": smallest.id, "ctx": MIN_CTX},
    }


# ------------------------------------------------------------------ .env -----

def read_env(path: Path) -> dict[str, str]:
    out = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.split("#", 1)[0].strip().strip('"').strip("'")
    return out


def _env_value(value: str) -> str:
    """One .env line's worth of value.

    .env is parsed line by line, so a newline inside a value silently becomes a
    second setting - and the values written here are not all ours: PROFILE_GPU
    comes from nvidia-smi, HF_TOKEN from a text box. Anything that could start a
    new line, or comment out the rest of this one, is dropped."""
    text = str(value if value is not None else "")
    text = text.replace("\r", " ").replace("\n", " ").replace("\0", "")
    text = "".join(ch for ch in text if ch >= " " or ch == "\t")
    return text.split("#", 1)[0].strip()


def write_env(path: Path, updates: dict[str, str]) -> None:
    """Set keys in .env in place (uncommented KEY=... lines), append the rest.
    Writes via a temp file + replace; clears a Windows read-only attribute if
    that is what blocks the write. Raises PermissionError if the file is locked."""
    import os, stat, tempfile
    updates = {k: _env_value(v) for k, v in updates.items()}
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    done = set()
    for i, raw in enumerate(lines):
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", raw)
        if m and m.group(1) in updates:
            lines[i] = f"{m.group(1)}={updates[m.group(1)]}"
            done.add(m.group(1))
    rest = [k for k in updates if k not in done]
    if rest:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# --- profile chosen by windows\\START-HERE.bat / linux/setup.sh (run `simplex setup` to change) ---")
        for k in rest:
            lines.append(f"{k}={updates[k]}")
    data = "\n".join(lines) + "\n"
    for attempt in range(2):
        try:
            fd, tmp = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=str(path.parent), text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, path)
            return
        except PermissionError:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            if attempt == 0 and path.is_file():
                try:
                    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)   # clear read-only attribute
                    continue
                except Exception:
                    pass
            raise


def env_updates(o: dict, gpu_name: str) -> dict[str, str]:
    model_id = Path(o["model_dir"]).name.lower()
    return {
        "PROFILE": f"{o['quant']}bpw-{round(o['ctx'] / 1000)}k",
        "PROFILE_GPU": gpu_name.replace("=", " ").split("  [")[0] or "unknown",
        "MODEL_DIR": o["model_dir"],
        "HF_TARGET_REPO": o["repo"],
        "HF_REVISION": o["revision"],
        "MODEL_ID": model_id,
        "CONTEXT_SIZE": str(o["ctx"]),
        "CACHE_QUANT": o["cache"],
        # cap = what this profile needs + 1.5 GB headroom, never above the card's budget;
        # the pre-flight then asks for that much free VRAM, not the whole card
        "GPU_MEM_GB": f"{min(o['budget'], o['need'] + 1.5):.1f}",
        "VISION": "auto" if o["vision"] else "off",
    }


# ------------------------------------------------------------------- UI ------

def timed_input(prompt: str, timeout: float) -> str | None:
    print(prompt, end="", flush=True)
    if sys.platform != "win32":
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            return sys.stdin.readline().strip()
        print()
        return None
    import msvcrt
    buf, t0 = [], time.time()
    while time.time() - t0 < timeout:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                print(); return "".join(buf).strip()
            if ch in ("\x00", "\xe0"):
                # An arrow or function key arrives as a two-character sequence:
                # this prefix, then a code that is itself printable ("H" for Up).
                # Dropping only the prefix let the second half land in the answer,
                # so pressing Up typed an H into it. Swallow both halves.
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            if ch == "\x08":
                if buf:
                    buf.pop(); print("\b \b", end="", flush=True)
            elif ch == "\x03":
                raise KeyboardInterrupt
            elif ch.isprintable():
                buf.append(ch); print(ch, end="", flush=True)
        else:
            time.sleep(0.05)
    print()
    return None


# The ladder the menu prints, worst to best: fair < good < better < very good <
# excellent < near-lossless. It has to stay in step with kl (lower = closer to
# the unquantised model) - a row labelled better than the row above it is worse
# than no label at all. test_quality_labels_do_not_invert() pins that.
QUALITY_WORD = {"near-lossless": "near-lossless", "excellent": "excellent",
                "very good": "very good", "better": "better", "good": "good",
                "fair": "fair", "fair (16 GB baseline)": "fair"}
QUALITY_ORDER = ["fair", "good", "better", "very good", "excellent", "near-lossless"]
COLS = "{n:>3}  {qual:<14} {ctx:<12} {img:<7} {vram:>10}  {dl:>10}   {quant:<8}{tag}"


def ctx_label(ctx: int) -> str:
    return "262k (max)" if ctx >= NATIVE_CTX else f"{round(ctx / 1000)}k"


def row(n, qual, ctx, img, vram, dl, quant, tag="") -> str:
    return COLS.format(n=n, qual=qual, ctx=ctx, img=img, vram=vram, dl=dl, quant=quant, tag=tag)


def show_menu(g: "GPU", budget: float, options: list[dict], current: dict | None) -> None:
    arch = f"  ({g.arch})" if g.cc else ""
    drv = f"  |  driver {g.driver}" if g.driver else ""
    print()
    print(f"  GPU: {g.name or 'unknown'}{arch}  |  {g.total_gib:.0f} GB VRAM{drv}")
    print(f"  Budget for the model: {budget:.1f} GB   (VRAM minus driver/desktop headroom)")
    print()
    print("  " + row("#", "Quality", "Context", "Images", "VRAM use", "Download", "Quant"))
    print("  " + "-" * 85)
    for i, o in enumerate(options, 1):
        # Two different things can be unknown about a row, so they are marked
        # separately rather than lumped into one vague warning:
        #   VRAM*    - the weights were never loaded on real hardware, the figure
        #              comes from the published shard sizes
        #   context* - no full-context prefill has ever been proven at this size,
        #              so it is the formula's answer, not an observed one
        # the marker goes inside the field, not after it: appending to a
        # right-aligned cell pushed marked rows one column out of line
        vram_cell = f"{o['need']:.1f} GB" + ("*" if not o.get("measured", True) else " ")
        ctx_cell = ctx_label(o["ctx"]) + ("" if o.get("verified_ctx") else "*")
        print("  " + row(i, QUALITY_WORD.get(o["note"], o["note"]), ctx_cell,
                         "yes" if o["vision"] else "no", vram_cell, f"{o['disk_gb']:.1f} GB",
                         f"{o['bpw']:.1f} bpw", "  <- recommended" if o["recommended"] else ""))
    if current:
        print("  " + row(0, "keep current", ctx_label(current["ctx"]), current["img"], "-", "installed",
                         current["quant"], "  (" + current["qual"] + ")" if current["qual"] != "-" else ""))
    print()
    if any(not o.get("measured", True) or not o.get("verified_ctx") for o in options):
        print("  * not verified on hardware yet: a VRAM* figure is from the published shard")
        print("    sizes, a context* is the formula's ceiling and no prefill has proven it.")
    print("  Quality  = how close to the unquantised model: "
          + " > ".join(reversed(QUALITY_ORDER)) + ".")
    print("  VRAM use = weights + KV cache for that context + images + runtime; mostly context on big cards.")
    print("  Higher quality = bigger download and slower tokens/s.   Change any time:  " +
          ("simplex setup" if sys.platform != "win32" else "windows\\START-HERE.bat"))
    print()


def choose(options: list[dict], auto: bool, default: int = 1, timeout: float = 90) -> dict | None:
    """options are numbered from 1; index 0 (if given) is 'keep current'. default is 1-based
    (0 = keep). Returns the chosen option dict, {"kind": "keep"} or None."""
    if auto or not options:
        if default == 0:
            return {"kind": "keep"}
        return options[default - 1] if options else None
    lo = 0 if default == 0 else 1
    while True:
        hint = "keep current" if default == 0 else "recommended"
        ans = timed_input(f"  ? Choose {lo}-{len(options)}   [Enter = {default} ({hint}), q = quit, auto in {timeout:.0f}s]: ", timeout)
        if ans is None or ans == "":
            ans = str(default)
        if ans.isdigit():
            n = int(ans)
            if n == 0 and lo == 0:
                return {"kind": "keep"}
            if 1 <= n <= len(options):
                return options[n - 1]
        if ans.lower() in ("q", "quit"):
            return None
        print("  type the number of a profile")


# ----------------------------------------------------- what is on disk ------
# windows\start.bat starts a model; windows\START-HERE.bat puts models on the disk. So the start
# side never asks the download questions - it asks the one question a machine
# with several weights on it actually raises, which is which of them to load.

INST_COLS = "{n:>3}  {qual:<14} {ctx:<12} {img:<7} {vram:>10}   {quant:<8}{tag}"


def installed(total_gib: float, cfg: dict | None = None) -> list[dict]:
    """Every quant whose weights are complete on this disk, best quality first.

    A row is a whole profile - the same dict plan() builds - so starting one
    goes through the same code as choosing one during setup: context, cache and
    the VRAM budget are planned for the card that is in the machine now, rather
    than read back from whatever .env was last left saying. The row that
    matches .env is the exception; it keeps .env's own numbers, because those
    may have been tuned by hand and a start is no place to overwrite them.
    """
    cfg = cfg or {}
    budget = budget_gib(total_gib) if total_gib > 0 else 0.0
    cur_dir = Path((cfg.get("MODEL_DIR") or "").replace("\\", "/")).name.lower()
    rows: list[dict] = []
    for q in QUANTS:
        st = on_disk(q.model_dir)
        if st["state"] != "complete":
            continue
        ctx, vision, fits = 0, False, False
        if budget > 0:
            picked = plan_one(q, "4", budget)
            if picked is not None:
                ctx, vision, fits = picked[0], picked[1], True
        is_current = Path(q.model_dir).name.lower() == cur_dir and bool(cur_dir)
        if is_current:
            try:
                env_ctx = int(cfg.get("CONTEXT_SIZE") or 0)
            except ValueError:
                env_ctx = 0
            if env_ctx > 0:
                ctx, fits = env_ctx, True
                vision = (cfg.get("VISION") or "auto").strip().lower() \
                    not in ("off", "0", "false", "no", "none")
        need = round(need_for(q, ctx, "4", vision, budget), 1) if ctx else 0.0
        rows.append({
            "quant": q.id, "bpw": q.bpw, "repo": q.repo, "revision": q.revision,
            "model_dir": q.model_dir, "ctx": ctx, "cache": "4", "vision": vision,
            "need": need, "disk_gb": q.disk_gb, "kl": q.kl, "note": q.note,
            "budget": budget, "recommended": False, "measured": q.measured,
            "verified_ctx": verified_ctx_for(q, vision),
            "verified_budget": q.verified_budget,
            "current": is_current, "fits": fits,
            "tight": bool(ctx) and need > budget > 0, "bytes": st["bytes"],
        })
    rec = recommend([o for o in rows if o["fits"]])
    if rec is not None:
        rec["recommended"] = True
    return rows


def unfinished() -> list[dict]:
    """Quants with something on disk but not all of it - a download that was
    stopped. They are listed under the menu, not in it: they cannot be started,
    and saying so is better than leaving the user to wonder where 3.0bpw went."""
    out = []
    for q in QUANTS:
        st = on_disk(q.model_dir)
        if st["state"] == "partial":
            out.append({"quant": q.id, "note": q.note, "model_dir": q.model_dir,
                        "disk_gb": q.disk_gb, "bytes": st["bytes"],
                        "reason": st["reason"]})
    return out


def show_installed(g: "GPU", rows: list[dict], pending: list[dict],
                   keep: str = "") -> None:
    budget = rows[0]["budget"] if rows else 0.0
    arch = f"  ({g.arch})" if g.cc else ""
    print()
    print(f"  Models on this PC   |   {g.name or 'unknown GPU'}{arch}"
          + (f"  |  {budget:.1f} GB usable" if budget else ""))
    print()
    print("  " + INST_COLS.format(n="#", qual="Quality", ctx="Context", img="Images",
                                  vram="VRAM use", quant="Quant", tag=""))
    print("  " + "-" * 74)
    for i, o in enumerate(rows, 1):
        if not o["fits"]:
            vram_cell, ctx_cell = "-", "does not fit"
        else:
            vram_cell = f"{o['need']:.1f} GB" + ("*" if not o.get("measured", True) else " ")
            ctx_cell = ctx_label(o["ctx"]) + ("" if o.get("verified_ctx") else "*")
        tag = "  <- last used" if o["current"] else (
            "  <- recommended" if o["recommended"] else "")
        if o["tight"]:
            tag += "  (over the budget)"
        print("  " + INST_COLS.format(
            n=i, qual=QUALITY_WORD.get(o["note"], o["note"]), ctx=ctx_cell,
            img="yes" if o["vision"] else "no", vram=vram_cell,
            quant=f"{o['bpw']:.1f} bpw", tag=tag))
    if keep:
        print("  " + INST_COLS.format(n=0, qual="keep current", ctx="from .env",
                                      img="-", vram="-", quant=Path(keep).name[:8],
                                      tag="  <- not one of the sizes above"))
    for o in pending:
        got = o["bytes"] / 1000 ** 3
        print(f"    -  {QUALITY_WORD.get(o['note'], o['note']):<14} "
              f"{o['quant']} bpw is {got:.1f} of {o['disk_gb']:.1f} GB "
              f"- unfinished download")
    print()
    if any(not o.get("measured", True) or not o.get("verified_ctx") for o in rows if o["fits"]):
        print("  * not verified on hardware yet: a VRAM* figure is from the published")
        print("    shard sizes, a context* is the formula's ceiling.")
    if pending:
        print("  An unfinished download is resumed by setup, not by a start.")
    print()


def choose_installed(cfg: dict, auto: bool = False, timeout: float = 45.0,
                     g: "GPU | None" = None) -> dict | str | None:
    """Ask which of the downloaded models to start.

    Returns the chosen profile dict, or one of two strings: "setup" when the
    answer is "fetch another size" (also what nothing-on-disk returns, without
    asking - there is nothing to choose between), and "keep" when the question
    cannot be put sensibly and .env's own settings should stand.
    """
    g = g or detect_gpu()
    if g.total_gib <= 0:
        # nvidia-smi could not be read - a driver update mid-session, a
        # container, an SSH session without the tools. Every row would have to
        # be planned against a budget of zero and would read "does not fit",
        # which is worse than not asking: .env already holds settings that were
        # written when the card could be read, so start those.
        return "keep"
    rows = installed(g.total_gib, cfg)
    pending = unfinished()
    startable = [o for o in rows if o["fits"]]
    # A MODEL_DIR of the user's own - a quant this kit does not ship, or a
    # folder somewhere else entirely - is not in the table and must not be
    # thrown away by pressing Enter. It gets row 0, and row 0 is the default.
    cur_dir = (cfg.get("MODEL_DIR") or "").strip()
    keep = cur_dir if (cur_dir and on_disk(cur_dir)["state"] == "complete"
                       and not any(o["current"] for o in rows)) else ""
    if not startable:
        return "keep" if keep else "setup"
    show_installed(g, rows, pending, keep)
    # Enter starts what .env last started; when that model is not startable -
    # a fresh kit, or the last one was the half-downloaded one - Enter falls to
    # the same row setup would have recommended, never simply to the top of the
    # table (the top row is the biggest quant, and on a roomy card that is the
    # one whose full-context prefill has never been proven).
    # None, not 0: row 0 is "keep the model .env names", a real answer, and
    # testing it with `if not default` quietly threw it away for the
    # recommendation - which is the one thing a custom MODEL_DIR must not lose.
    default = next((i for i, o in enumerate(rows, 1) if o["current"] and o["fits"]), None)
    if default is None and keep:
        default = 0
    if default is None:
        default = next((i for i, o in enumerate(rows, 1) if o["recommended"]),
                       next(i for i, o in enumerate(rows, 1) if o["fits"]))
    if auto or len(rows) == 1 and not pending and rows[0]["current"]:
        return "keep" if default == 0 else rows[default - 1]
    while True:
        ans = timed_input(
            f"  ? Which one do you want to start?   [Enter = {default}, "
            f"s = setup, q = quit, auto in {timeout:.0f}s]: ", timeout)
        if ans is None or ans.strip() == "":
            return "keep" if default == 0 else rows[default - 1]
        ans = ans.strip().lower()
        if ans.isdigit():
            n = int(ans)
            if n == 0 and keep:
                return "keep"
            if 1 <= n <= len(rows):
                if rows[n - 1]["fits"]:
                    return rows[n - 1]
                print("  that one does not fit this card - it would run out of VRAM")
                continue
        if ans in ("s", "setup"):
            return "setup"
        if ans in ("q", "quit"):
            return None
        print(f"  type the number of a model ({'0' if keep else '1'}-{len(rows)}), "
              "s for setup, or q to quit")


def vision_cost_line(total_gib: float) -> str:
    """What the tower actually costs on this card, in tokens of context.

    Quoting one figure was wrong: the tower is 0.87 GiB on the 3.0-6.0bpw quants
    but only 0.17 GiB on the 2.0bpw build (its own is quantised to 3 bits), and
    2.0bpw is exactly what a 16 GB card is steered to - so the old line overstated
    the trade fivefold on the card class this kit exists for. Quote the range that
    actually applies to the quants this card can hold."""
    budget = budget_gib(total_gib)
    fits = [q for q in QUANTS if max_ctx(q, "4", True, budget) >= MIN_CTX]
    if not fits:
        fits = list(QUANTS)
    per_token_gib = (KV_KB["4"] * MTP_FACTOR + OVERHEAD_KB_PER_TOKEN) * 1024 / 1024 ** 3
    costs = sorted({q.vision_gib for q in fits})
    lo, hi = costs[0], costs[-1]
    lo_tok, hi_tok = int(lo / per_token_gib / 1000), int(hi / per_token_gib / 1000)
    if abs(hi - lo) < 0.05:
        return f"about {hi:.1f} GB of VRAM - roughly {hi_tok}k tokens of context"
    return (f"{lo:.1f}-{hi:.1f} GB of VRAM - "
            f"roughly {lo_tok}k-{hi_tok}k tokens of context")


def ask_start(cfg: dict, auto: bool = False) -> str | None:
    """Ask what a start should start: the whole kit, or only the model.

    UI=no serves /v1 and nothing else - no chat page, and no Agent that can
    read and write files here and run commands. That was always a setting in
    .env, but nothing asked, so the first anyone knew of it was a browser
    opening by itself. The browser setup page asks the same question; this is
    the console's copy of it, so the two front doors to one install agree.

    Returns None when nobody answered - the timeout, or a closed stdin - and
    the caller leaves .env exactly as it was.
    """
    if auto:
        return None
    current = (cfg.get("UI") or "browser").strip().lower()
    print()
    print("  The model always serves /v1 for other apps.")
    print("  The rest of the kit is the DeepSeek Harness: a chat and agent")
    print("  front end that runs on this PC and is configured for this model")
    print("  automatically. It needs Node installed.")
    default = "n" if current == "no" else "y"
    ans = timed_input(
        f"  ? Start the DeepSeek Harness too?  [Enter = {default}, y / n, auto in 60s]: ", 60)
    if ans is None:
        return None
    ans = ans.strip().lower() or default
    return "no" if ans.startswith("n") else "browser"


def ask_vision(auto: bool = False, total_gib: float = 0.0) -> bool | None:
    """Ask whether this install needs image input, before the menu is built.

    The vision tower sits on the card whether or not a picture is ever pasted,
    and that VRAM would otherwise be context. Only the person installing knows
    which side of that trade they want, so ask instead of guessing.

    Returns None when nobody answered, which plan() reads as "use the automatic
    rule" - the old behaviour, unchanged. Both ways of not answering have to lead
    there: the 60s timeout, and a closed or redirected stdin. EOF used to fall
    through to `not "".startswith("n")` and force images ON, so a launcher started
    with stdin from NUL, or a scheduled run, silently took the narrower menu."""
    if auto:
        return None
    print()
    print("  Images: this model can read pictures you paste into the chat.")
    print(f"  It costs {vision_cost_line(total_gib)}.")
    print("  Text-only frees that up.")
    ans = timed_input("  ? Do you want image input?  [Enter = yes, n = no, auto in 60s]: ", 60)
    if ans is None:
        return None                      # timed out: nobody there
    ans = ans.strip().lower()
    if not ans:
        # Enter means yes; EOF also arrives as "" but is not an answer, so tell
        # them apart by whether stdin is still a live terminal to answer from.
        if not sys.stdin.isatty():
            return None
        return True
    return not ans.startswith("n")


def ask_vram() -> float:
    ans = timed_input("  ? Could not read the GPU with nvidia-smi. VRAM in GB (e.g. 16, 24): ", 120)
    try:
        return float(ans) if ans else 16.0
    except ValueError:
        return 16.0


# ----------------------------------------------------------------- main ------

def run(force: bool = False, auto: bool = False, vram: float | None = None, list_only: bool = False,
        cc_override: float | None = None) -> int:
    cfg = read_env(ENV_FILE)
    if not force and not list_only and cfg.get("PROFILE") and cfg["PROFILE"].lower() != "ask":
        return 0   # already chosen
    g = detect_gpu()
    if vram:
        g.total_gib = float(vram)
        g.name = g.name or f"(assumed {vram} GB)"
    if cc_override:
        g.cc = cc_override
    total = g.total_gib
    if total <= 0:
        if auto or list_only:
            total = 16.0
        else:
            total = ask_vram()
        g.total_gib = total
    status, notes = gpu_support(g)
    gpu_name = g.name
    for n in notes:
        print(f"  ! {n}")
    if status == "unsupported":
        print("  Stopping: this GPU cannot run the kit.")
        return 3
    # Ask about images first: it changes which quants fit and how much context each
    # one gets, so the menu below is built for the answer rather than guessing per row.
    want_vision = None if (auto or list_only) else ask_vision(auto, total)
    budget, options = plan(total, want_vision)
    if want_vision:
        text_only = plan(total, False)[1]
        shown = {o["quant"] for o in options}
        hidden = [o for o in text_only if o["quant"] not in shown]
        if hidden and options:
            # These are the *largest* quants - the ones whose weights leave no room
            # for the tower - so calling them "lower-VRAM" pointed the user away
            # from the best model their card can run.
            names = ", ".join(f"{o['quant']}bpw" for o in hidden)
            print(f"  ! Hidden because they cannot fit images and {MIN_CTX // 1024}k "
                  f"context on this card: {names}")
            print("    (higher quality than the rows below - answer n to images to use them)")
        elif hidden and not options:
            # Every quant that fits was dropped by the images requirement, so the
            # card is not too small - the answer was. Saying "the 27B does not fit"
            # here names the wrong cause and aborts an install that would work.
            best = max(hidden, key=lambda o: o["ctx"])
            print(f"  ! Nothing fits with images on this {total:.0f} GB card, but "
                  f"{best['quant']}bpw runs text-only")
            print(f"    at {ctx_label(best['ctx'])} context. Showing the text-only options instead.")
            options = text_only
            want_vision = False
    # an existing, downloaded configuration is offered as "keep" and is the timeout default,
    # so an unattended start never triggers a surprise multi-GB download
    cur_dir = cfg.get("MODEL_DIR", "")
    current = None
    if cur_dir and on_disk(cur_dir)["state"] == "complete":
        m = re.search(r"(\d\.\d)bpw", Path(cur_dir).name)
        qid = m.group(1) if m else ""
        qrow = next((q for q in QUANTS if q.id == qid), None)
        try:
            cur_ctx = int(cfg.get("CONTEXT_SIZE", "0"))
        except ValueError:
            cur_ctx = 0
        current = {"quant": f"{qid} bpw" if qid else Path(cur_dir).name[:8], "ctx": cur_ctx,
                   "img": "no" if (cfg.get("VISION", "auto").lower() in ("off", "0", "false", "no")) else "yes",
                   "qual": QUALITY_WORD.get(qrow.note, qrow.note) if qrow else "-"}
    show_menu(g, budget, options, current)
    if list_only:
        return 0
    if not options:
        print(f"  ! No profile fits in {total:.0f} GB of VRAM (the 27B needs about "
              f"{min(q.gpu_gib for q in QUANTS) + OVERHEAD_GIB + MARGIN_GIB:.0f} GB "
              "before any context). Keeping the .env as is.")
        return 1
    rec_index = next((i + 1 for i, o in enumerate(options) if o.get("recommended")), 1)
    default = 0 if current else rec_index
    o = choose(options, auto, default)
    if o is None:
        print("  no profile chosen")
        return 1
    if o.get("kind") == "keep":
        if not cfg.get("PROFILE") or cfg["PROFILE"].lower() == "ask":
            try:
                write_env(ENV_FILE, {"PROFILE": "current", "PROFILE_GPU": gpu_name or "unknown"})
            except PermissionError:
                print("  ! .env is locked or read-only (another program has it open?) - could not note the")
                print("    choice; this menu will show again next start. Add PROFILE=current to .env to stop it.")
        print("  OK  keeping the current settings")
        return 0
    upd = env_updates(o, gpu_name)
    # asked here as well as on the setup page, so the console front door and
    # the browser one write the same file for the same reasons
    want_ui = ask_start(cfg, auto)
    if want_ui is not None:
        upd["UI"] = want_ui
    try:
        write_env(ENV_FILE, upd)
    except PermissionError:
        print("  ! Cannot write .env: it is locked or read-only (an editor or sync tool has it open,")
        print("    or the file's read-only attribute is set). Close it / clear the attribute and run again.")
        return 4
    print(f"  OK  profile {upd['PROFILE']} written to .env  ({o['bpw']:.1f} bpw, {o['ctx']} tokens, KV {o['cache']})")
    st = on_disk(o["model_dir"])
    if st["state"] == "missing":
        print(f"      first start downloads {o['disk_gb']:.1f} GB into {o['model_dir']}")
    elif st["state"] == "partial":
        got = st["bytes"] / 1000 ** 3
        print(f"      {o['model_dir']} is {got:.1f} of {o['disk_gb']:.1f} GB - "
              f"setup resumes it ({st['reason']})")
    return 0


def pick(auto: bool = False) -> int:
    """The start-time question, as a command: which of the downloaded models?

    Writes the answer into .env. Exit codes are the answer, because the shell
    launcher is the caller:
        0  a model is chosen and .env says so
        2  nothing is startable - the answer is setup, not a model
        3  quit; nobody asked for anything to start
    """
    cfg = read_env(ENV_FILE)
    g = detect_gpu()
    choice = choose_installed(cfg, auto=auto, g=g)
    if choice is None:
        return 3
    if choice == "setup":
        return 2
    if choice == "keep":
        return 0
    if choice["current"] and (cfg.get("PROFILE") or "").strip():
        return 0                          # already what .env says; leave it alone
    try:
        write_env(ENV_FILE, env_updates(choice, g.name))
    except PermissionError:
        print("  ! Cannot write .env: another program has it open.")
        return 4
    print(f"  OK  starting {Path(choice['model_dir']).name}  "
          f"({choice['bpw']:.1f} bpw, {choice['ctx']} tokens, "
          f"images {'on' if choice['vision'] else 'off'})")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="ask even if .env already has a PROFILE")
    ap.add_argument("--auto", action="store_true", help="take the recommended option without asking")
    ap.add_argument("--vram", type=float, help="override detected VRAM (GB)")
    ap.add_argument("--list", action="store_true", help="show the options and exit")
    ap.add_argument("--cc", type=float, help="override detected compute capability (e.g. 7.5)")
    ap.add_argument("--pick", action="store_true",
                    help="ask which downloaded model to start and write it to .env")
    a = ap.parse_args(argv[1:])
    try:
        if a.pick:
            return pick(auto=a.auto)
        return run(force=a.force, auto=a.auto, vram=a.vram, list_only=a.list, cc_override=a.cc)
    except KeyboardInterrupt:
        print("\n  interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv))

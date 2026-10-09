#!/usr/bin/env python3
"""DRAFT=dflash2: the opt-in DFlash2 drafter. Everything it needs to know in one place.

The default drafter is the MTP head inside the checkpoint. DRAFT=dflash2 swaps
it for a separate drafter model. Both run on ExLlamaV3 1.6.0 in the one .venv,
so changing DRAFT back and forth never reinstalls anything. The two launchers (linux/start.sh, tools/win_start.py) and
tools/setup_core.py all ask this file instead of each carrying a copy.

What was measured, and so what is allowed (README, "DFlash2"): one RTX 5090 on
Linux, the kit's own 4.0 bpw quant at 262144 tokens, `--grid_size 22.8
--cache_quant 4`, images on. That is the only setting this accepts. On a 32 GB
card the profile planner already writes the 4.0 bpw row as exactly that; the
launchers still apply the measured numbers for the run, so a hand-edited .env
cannot drift from what was measured (nothing is written to .env).

Standard library only (it runs before the virtualenv exists):

    python tools/dflash2.py gate                    refuse a card that is too small
    python tools/dflash2.py pin --model-dir DIR     refuse an unmeasured quant, else
                                                    print the settings to run with
    python tools/dflash2.py verify DIR              check the drafter's checksum
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DRAFT = "dflash2"
ENGINE_VERSION = "1.6.0"
VENV_NAME = ".venv"

# Installed next to the kit's own server libraries. 1.4.4 got transformers
# through its flash-linear-attention requirement; 1.6.0 carries those kernels
# itself and no longer pulls it in, but its chat template
# (Tokenizer.hf_chat_template) still imports it - without it every chat request
# returns HTTP 500. Needed for every DRAFT value, not only dflash2.
EXTRA_PACKAGES = ("transformers",)

# The drafter: an EXL3 4.00 bpw quant of z-lab's DFlash2 (r0b0tlab's upload).
# Pinned to a commit rather than a branch, and to the checksum of its one weight
# file, so what is downloaded is what was measured.
DRAFT_REPO = "r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw"
DRAFT_REVISION = "265b5240592907d2d55ff0dc4d5f66569692604d"
DRAFT_DIR = "models/Qwen3.8-27B-DFlash2-EXL3-4.00bpw"
DRAFT_FILE = "model.safetensors"
DRAFT_SHA256 = "e278f218318565562af07fc333045e411ffa0d83523f8224fc8c2d24d5a68223"

# Total VRAM below which it is refused. Measured on a 32 GB RTX 5090 the server
# process held 24850 MiB (4.0 bpw) with the drafter, which does not fit a 24 GB
# card.
MIN_VRAM_GIB = 30.0

# The measured rows: quant id -> CONTEXT_SIZE. Nothing else is enabled. 5.0 bpw
# was not measured on the kit's own 5.0 file, so it is not enabled.
MEASURED = {"4.0": 262144}
GRID_GB = "22.8"
CACHE_QUANT = "4"


def wanted(draft: str | None) -> bool:
    return (draft or "").strip().lower() == DRAFT


def venv_name(draft: str | None = None) -> str:
    return VENV_NAME


def engine_version(draft: str | None = None, default: str = ENGINE_VERSION) -> str:
    return ENGINE_VERSION


def extra_packages(draft: str | None = None) -> tuple[str, ...]:
    return EXTRA_PACKAGES


def gate_message(total_gib: float) -> str:
    """Why this card is refused, or "" when it is not."""
    if total_gib <= 0:
        return ("DRAFT=dflash2 needs a 32 GB class NVIDIA card and nvidia-smi did not report "
                "one.\nSet DRAFT=mtp (the default) in .env to run without it.")
    if total_gib < MIN_VRAM_GIB:
        return (f"DRAFT=dflash2 needs a 32 GB class card (at least {MIN_VRAM_GIB:.0f} GiB); "
                f"this one has {total_gib:.1f} GiB.\n"
                "On a 32 GB RTX 5090 the server held 24850 MiB (4.0 bpw) with the "
                "drafter,\nwhich does not fit a 24 GB card. "
                "Set DRAFT=mtp (the default) in .env to run without it.")
    return ""


def pin(model_dir: str) -> tuple[dict[str, str] | None, str]:
    """(settings, "") for a measured quant, else (None, why not).

    The settings are the ones the speed numbers were measured with, for this run
    only: the launchers apply them over .env without writing them back."""
    import profiles
    name = Path((model_dir or "").replace("\\", "/")).name.lower()
    q = next((q for q in profiles.QUANTS if Path(q.model_dir).name.lower() == name), None)
    if q is None or q.id not in MEASURED:
        return None, (f"DRAFT=dflash2 was only measured with the kit's 4.0 bpw quant "
                      f"(MODEL_DIR is {model_dir or 'not set'}).\n"
                      "Pick that one in setup, or set DRAFT=mtp (the default) in .env.")
    return {"CONTEXT_SIZE": str(MEASURED[q.id]), "GPU_MEM_GB": GRID_GB,
            "CACHE_QUANT": CACHE_QUANT}, ""


def verify(dest: Path) -> str:
    """"" when the drafter's weight file is the pinned one, else what is wrong."""
    f = Path(dest) / DRAFT_FILE
    if not f.is_file():
        return f"{f} is missing."
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != DRAFT_SHA256:
        return (f"{f} does not match the pinned checksum ({DRAFT_SHA256}).\n"
                f"Delete the folder {Path(dest)} and start again to download it afresh.")
    return ""


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="exit 1 unless the card is big enough")
    g.add_argument("--vram", type=float, help="pretend the card has this many GiB")
    p = sub.add_parser("pin", help="print the settings to run a measured quant with")
    p.add_argument("--model-dir", required=True)
    v = sub.add_parser("verify", help="check the drafter's checksum")
    v.add_argument("dir")
    a = ap.parse_args(argv)

    if a.cmd == "gate":
        import profiles
        why = gate_message(a.vram if a.vram is not None else profiles.detect_gpu().total_gib)
    elif a.cmd == "pin":
        settings, why = pin(a.model_dir)
        if settings is not None:
            for k, val in {**settings, "DFLASH2_REPO": DRAFT_REPO, "DFLASH2_REVISION": DRAFT_REVISION,
                           "DFLASH2_DIR": DRAFT_DIR}.items():
                print(f"{k}={val}")
    else:
        why = verify(Path(a.dir))
    if why:
        print(f"\n  ERROR  {why}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env bash
# quantize.sh — rebuild the 2.00bpw EXL3 quant from the unquantized HF model
# using recipe_2.00bpw.yaml (the per-tensor bitrate recipe in this repo).
#
# Requires ExLlamaV3 v1.4.4 (same venv start.sh creates).
#
# Usage:
#   IN_DIR=/models/qwen3.8-27b-hf ./quantize.sh
#   WORK_DIR=/tmp/exl3-work-2.00bpw ./quantize.sh -r     # resume a job
#
# Why -mb / -vb are passed explicitly: convert.py reads ONLY tensors,
# target_bpw / achieved_bpw and head_bits out of a recipe file. mtp_bits and
# vision_bits are carried in recipe_2.00bpw.yaml purely as documentation, so
# reproducing the H3 / V3 layout (head 3, MTP 2, vision 3) means passing
# -mb 2 -vb 3 on the command line. head_bits DOES come from the recipe.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && source .env
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

RESUME=0
for a in "$@"; do
    case "$a" in -r|--resume) RESUME=1 ;; esac
done

OUT_DIR="${OUT_DIR:-models/Qwen3.8-27B-EXL3-2.0bpw-build}"
WORK_DIR="${WORK_DIR:-/tmp/exl3-work-2.00bpw}"
RECIPE="${RECIPE:-recipe_2.00bpw.yaml}"
# cal_trace.safetensors ships in this repo: the self-calibration trace this
# quant was built on. Passing it is what makes this a self-calibrated quant.
CAL_DATA="${CAL_DATA:-cal_trace.safetensors}"

# --- preflight ---------------------------------------------------------------
if [ ! -f "$RECIPE" ]; then
    echo "ERROR: recipe not found: $RECIPE" >&2
    exit 1
fi

if [ "$RESUME" -eq 0 ]; then
    if [ -z "${IN_DIR:-}" ]; then
        echo "ERROR: IN_DIR must point at the unquantized Qwen3.8-27B HF directory" >&2
        echo "Fetch it with: hf download Qwen/Qwen3.8-27B --local-dir <dir>" >&2
        exit 1
    fi
    if [ ! -d "$IN_DIR" ]; then
        echo "ERROR: IN_DIR is not a directory: $IN_DIR" >&2
        exit 1
    fi
fi

_ver="$("$PYTHON" -c 'from exllamav3.version import __version__; print(__version__)' 2>/dev/null || echo missing)"
if [ "$_ver" != "1.4.4" ]; then
    echo "ERROR: quantize.sh needs exllamav3 1.4.4, found '$_ver'." >&2
    echo "Run ./start.sh once (it pins the v1.4.4 tag), or set PYTHON to a venv that has it." >&2
    exit 1
fi

cmd=("$PYTHON" -m exllamav3.conversion.convert_model)
if [ "$RESUME" -eq 1 ]; then
    # Resume: input/output/recipe/bitrates are restored from the job state.
    cmd+=(-w "$WORK_DIR" "$@")
else
    cmd+=(-i "$IN_DIR" -o "$OUT_DIR" -w "$WORK_DIR" -rcp "$RECIPE" -mb 2 -vb 3)
    if [ -f "$CAL_DATA" ]; then
        cmd+=(-cd "$CAL_DATA")
    else
        echo "note: $CAL_DATA not found -- falling back to exllamav3's bundled corpus mix."
        echo "      The result will NOT be the self-calibrated quant."
    fi
    cmd+=("$@")
fi

echo " engine : exllamav3 $_ver"
echo " recipe : $RECIPE (head_bits from recipe; -mb 2 -vb 3 on the CLI)"
echo " work   : $WORK_DIR"
echo " cmd    : ${cmd[*]}"
echo
exec "${cmd[@]}"

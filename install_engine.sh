#!/usr/bin/env bash
# install_engine.sh — build & install ExLlamaV3 v1.4.4 for this host.
#
# Why this exists: upstream exllamav3 v1.4.4 (and v1.4.5) CANNOT build on
# aarch64. setup.py globs every source under exllamav3_ext/ with no arch
# filter, and 7 of them are x86-only (avx512_target.cpp, avx2_target.cpp,
# cpu/moe_mul1.cpp, cpu/moe_handoff.cu, parallel/all_reduce_cpu*.cpp/cu —
# __builtin_cpu_supports, __builtin_ia32_pause, immintrin.h). On GB10 / DGX
# Spark the wheel build dies in those files.
#
# patches/aarch64-v1.4.4.patch excludes those 7 sources on non-x86 hosts and
# guards their pybind registrations. They only implement CPU GEMV / MoE expert
# offload and the x86 CPU tensor-parallel all-reduce — none of which GPU
# inference touches, and this model is dense (no experts) on a single GPU.
# On x86 the patch is a no-op (the guards compile the code back in).
#
# The MiaAI-Lab fork does build on GB10, but it is version 1.4.2, which is too
# old for this quant's quantized vision tower — so it is not an option here.
#
# Usage:  ./install_engine.sh            # into ./.venv
#         ENGINE_DIR=/path ./install_engine.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && source .env

PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
TAG="${EXL3_TAG:-v1.4.4}"
ENGINE_DIR="${ENGINE_DIR:-$(cd .. && pwd)/exllamav3-1.4.4-aarch64}"

if [ -d "$ENGINE_DIR/.git" ]; then
    echo "engine checkout already present: $ENGINE_DIR"
else
    echo "cloning exllamav3 $TAG -> $ENGINE_DIR"
    git clone --depth 1 --branch "$TAG" https://github.com/turboderp-org/exllamav3 "$ENGINE_DIR"
    git -C "$ENGINE_DIR" apply "$PWD/patches/aarch64-v1.4.4.patch"
    echo "applied patches/aarch64-v1.4.4.patch"
fi

# GB10/Spark needs the arch list spelled out; x86 auto-detects.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ] && [ "$(uname -m)" = "aarch64" ]; then
    export TORCH_CUDA_ARCH_LIST="12.0;12.1"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) >= 8 && $(free -g | awk '/^Mem:/{print $2}') >= 32 ? 8 : 4 ))}"
export PATH="$(dirname "$PYTHON"):$PATH"   # ninja lives in the venv

exec "$PYTHON" -m pip install --no-build-isolation "$ENGINE_DIR"

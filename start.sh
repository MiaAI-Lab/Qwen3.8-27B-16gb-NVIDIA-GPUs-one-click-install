#!/usr/bin/env bash
# Start the OpenAI-compatible exllamav3 server (tools/serve_openai.py).
# Configuration lives in .env — created from .env.example on first run.
#
# Works from the deployment kit or from the engine repo itself. First run
# builds .venv, installs torch + the engine (compiling the CUDA kernels) +
# server deps, then downloads the model weights from Hugging Face and serves.
# Later runs start the server directly.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f tools/serve_openai.py ]; then
    echo "start.sh must run from the deployment kit or the engine repo" >&2
    echo "(tools/serve_openai.py not found next to it)." >&2
    exit 1
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "No .env found — created one from .env.example."
    echo "Edit it (model paths, context, GPU memory) and run ./start.sh again."
    exit 1
fi
# shellcheck disable=SC1091
# GPU-aware profile (tools/profiles.py): fresh .env, PROFILE=ask, or `./start.sh profile`
# -> detect VRAM, choose quant / context / KV cache, write the choice into .env.
_pyprof="$(command -v python3 || command -v python || true)"
if [ -n "$_pyprof" ]; then
    case "${1:-}" in
        profile|--profile) "$_pyprof" tools/profiles.py --force || exit 1 ;;
        *)
            if ! grep -q '^PROFILE=' .env || grep -qi '^PROFILE=ask' .env; then
                "$_pyprof" tools/profiles.py || exit 1
            fi ;;
    esac
fi

source .env

# .env is sourced as shell vars; the model-download subprocess needs the HF
# token in its environment, so export it if set.
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi

# --- bootstrap: build the venv + install the engine on first run ----------
# Re-enters if the venv is missing OR the install is incomplete (e.g. a
# Ctrl-C during the first run left a half-installed venv) — pip is idempotent.
if [ ! -x .venv/bin/python ] \
   || ! .venv/bin/python -c "import torch, exllamav3, aiohttp, huggingface_hub" 2>/dev/null; then
    echo "First-run setup — one time only (later runs skip straight to the model):"
    BOOT_LOG="$(mktemp /tmp/exl3_setup.XXXXXX.log)"

    _elapsed() { printf '%dm%02ds' $(($1 / 60)) $(($1 % 60)); }

    # Step whose own output is useful (pip download bars): run in foreground.
    _step() {   # _step "label" cmd [args…]
        local label="$1" t0=$SECONDS; shift
        echo "  [ .. ] $label"
        if "$@"; then
            echo "  [ ok ] $label ($(_elapsed $((SECONDS - t0))))"
        else
            echo "  [FAIL] $label — after $(_elapsed $((SECONDS - t0)))"
            return 1
        fi
    }

    # Long silent step (CUDA compile): spinner + live timer on a terminal,
    # plain lines when piped; output captured, tail shown on failure.
    _quiet_step() {   # _quiet_step "label" cmd [args…]
        local label="$1" t0=$SECONDS; shift
        : > "$BOOT_LOG"
        if [ -t 1 ]; then
            "$@" >>"$BOOT_LOG" 2>&1 &
            local pid=$! i=0 spin='-\|/'
            while kill -0 "$pid" 2>/dev/null; do
                prog=$(grep -oE '^\[[0-9]+/[0-9]+\]' "$BOOT_LOG" 2>/dev/null | tail -1 || true)
                printf '\r  [%s] %s … %s %s   ' "${spin:$((i % 4)):1}" "$label" \
                    "${prog:+$prog }" "$(_elapsed $((SECONDS - t0)))"
                i=$((i + 1)); sleep 0.25
            done
            if wait "$pid"; then
                printf '\r\033[K  [ ok ] %s (%s)\n' "$label" "$(_elapsed $((SECONDS - t0)))"
                return 0
            fi
        else
            echo "  [ .. ] $label"
            if "$@" >>"$BOOT_LOG" 2>&1; then
                echo "  [ ok ] $label ($(_elapsed $((SECONDS - t0))))"
                return 0
            fi
        fi
        printf '\r\033[K  [FAIL] %s — after %s\n' "$label" "$(_elapsed $((SECONDS - t0)))"
        echo "  ---- last output (full log: $BOOT_LOG) ----"
        tail -n 20 "$BOOT_LOG" | sed 's/^/  | /'
        return 1
    }

    _step "1/5 creating Python virtualenv" python3 -m venv .venv
    _quiet_step "2/5 build tools (pip, setuptools, wheel)" \
        .venv/bin/pip install --quiet --upgrade pip setuptools wheel typing_extensions packaging
    # GPU torch + its NVIDIA runtime deps; PyPI stays primary so the
    # nvidia-* runtime wheels resolve too (cu130 local-version wheel wins).
    # Output NOT hidden: pip's own download progress bars show here.
    _step "3/5 PyTorch (~2–3 GB download the first time)" \
        .venv/bin/pip install torch \
            --extra-index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    # The engine itself; its setup.py pulls in the rest of the deps.
    # Inside the engine repo: build from the local checkout (EXL3_REPO is
    # ignored there). Elsewhere (deployment kit): install from EXL3_REPO —
    # default is the official turboderp exllamav3 (v1.4.4+ required for
    # quantized vision tower); override in .env for a local path.
    # --no-build-isolation + the env vars below compile the native ext at
    # install time (override via .env as needed).
    if [ -f exllamav3/__init__.py ]; then
        _engine_src="."
        _engine_note="local engine repo — compiling CUDA kernels"
    else
        # Pinned to the v1.4.4 tag. PyPI has no 1.4.4 wheel (it jumps
        # 1.4.2 -> 1.4.5), so the tag is the only way to get exactly v1.4.4,
        # which this quant needs (quantized vision tower).
        _engine_src="${EXL3_REPO:-git+https://github.com/turboderp-org/exllamav3.git@v1.4.4}"
        _engine_note="exllamav3 engine — clone + compile CUDA kernels"
    fi
    if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        export TORCH_CUDA_ARCH_LIST
    elif [ "$(uname -m)" = "aarch64" ]; then
        # GB10/Spark needs the arch list spelled out; x86 auto-detects.
        export TORCH_CUDA_ARCH_LIST="12.0;12.1"
    fi
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    # 8 parallel nvcc jobs when the machine can take it (halves wall time on
    # many-core boxes); 4 otherwise. Override in .env if needed.
    export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) >= 8 && $(free -g | awk '/^Mem:/{print $2}') >= 32 ? 8 : 4 ))}"
    # Fail fast (clear error) instead of hanging if the engine repo needs
    # auth (GIT_ASKPASS: proven on git 2.43 where GIT_TERMINAL_PROMPTS
    # alone does not suppress the credential prompt).
    export GIT_TERMINAL_PROMPTS=0
    export GIT_ASKPASS=/bin/true
    _quiet_step "4/5 ${_engine_note} (5–20 min depending on machine)" \
        .venv/bin/pip install --no-build-isolation \
            "${_engine_src}"
    _quiet_step "5/5 server dependencies (aiohttp, huggingface_hub)" \
        .venv/bin/pip install --quiet aiohttp huggingface_hub
    echo "Setup complete."
fi

PYTHON=.venv/bin/python
# venv tools (ninja, …) must stay findable for the engine's JIT fallback.
export PATH="$(pwd)/.venv/bin:$PATH"

# --- engine version guard ---------------------------------------------------
# v1.4.4 is mandatory: this quant ships a quantized vision tower (vision_bits 3),
# which older builds decode incorrectly, and stock v1.4.4 is what this kit is
# validated against.
if ! "$PYTHON" -c 'import sys; from exllamav3.version import __version__ as v; sys.exit(0 if v == "1.4.4" else (print(" !! unexpected exllamav3 version:", v) or 1))' 2>/dev/null; then
    _gotver="$("$PYTHON" -c 'from exllamav3.version import __version__; print(__version__)' 2>/dev/null || echo unknown)"
    echo "ERROR: this kit requires ExLlamaV3 v1.4.4, but the venv has '$_gotver'." >&2
    echo "Fix: EXL3_REPO=git+https://github.com/turboderp-org/exllamav3.git@v1.4.4 then rm -rf .venv && ./start.sh" >&2
    exit 1
fi

MODEL_DIR="${MODEL_DIR:?MODEL_DIR must be set in .env}"
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"
CONTEXT_SIZE="${CONTEXT_SIZE:-199936}"
if [ -n "${GPU_MEM_GB:-}" ]; then
    echo "GPU memory budget: ${GPU_MEM_GB} GB (from .env)"
else
    _vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1) || true
    if [ "${_vram:-0}" -gt 0 ] 2>/dev/null; then
        # discrete GPU: VRAM minus a little headroom (e.g. 24 GB card -> 22)
        GPU_MEM_GB=$(( _vram / 1024 - 2 ))
    else
        # GB10/unified memory (nvidia-smi reports no total): available system
        # RAM minus a reserve for the OS and anything else on the box
        GPU_MEM_GB=$(( $(free -g | awk '/^Mem:/{print $7}') - 16 ))
    fi
    if [ "$GPU_MEM_GB" -lt 8 ]; then GPU_MEM_GB=8; fi
    echo "GPU_MEM_GB not set — auto-detected budget: ${GPU_MEM_GB} GB (override in .env)"
fi
CACHE_QUANT="${CACHE_QUANT:-none}"
CPU_CACHE_GB="${CPU_CACHE_GB:-0}"

# --- speculative decoding method ---------------------------------------------
# DRAFT = mtp | none (see .env.example for the trade-offs).
DRAFT="${DRAFT:-mtp}"
DRAFT="$(echo "$DRAFT" | tr '[:upper:]' '[:lower:]')"

# --- auto-download from the Hub if missing --------------------
# Set HF_TOKEN=<token> in .env, or `hf auth login`.
HF_TARGET_REPO="${HF_TARGET_REPO:-Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw}"

dl_model() {   # dl_model <repo_id> <dir> <label>
    local repo="$1" dir="$2" label="$3"
    if [ -f "$dir/config.json" ] && compgen -G "$dir/*.safetensors" > /dev/null; then
        echo "$label: $dir already present — skipping download."
        return 0
    fi
    echo "$label: not found at $dir — downloading from huggingface.co/$repo …"
    mkdir -p "$dir"
    "$PYTHON" - "$repo" "$dir" "${HF_REVISION:-}" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
rev = sys.argv[3] or None
path = snapshot_download(repo_id = sys.argv[1], local_dir = sys.argv[2], revision = rev)
print(f"  downloaded -> {path}")
PYEOF
}

dl_model "$HF_TARGET_REPO" "$MODEL_DIR" "target model"

# Context beyond the native 262144 needs the YaRN config variant.
if [ "$CONTEXT_SIZE" -gt 262144 ] \
   && [ -f "$MODEL_DIR/config.yarn-1m.json" ] \
   && ! grep -q rope_scaling "$MODEL_DIR/config.json"; then
    cp "$MODEL_DIR/config.yarn-1m.json" "$MODEL_DIR/config.json"
    echo "CONTEXT_SIZE > 262k: switched $MODEL_DIR/config.json to the YaRN 1M variant."
fi

MODEL_ID="${MODEL_ID:-$(basename "$MODEL_DIR" | tr '[:upper:]' '[:lower:]')}"
cmd=("$PYTHON" -u tools/serve_openai.py
     --model "$MODEL_DIR"
     --model_id "$MODEL_ID"
     --host "$HOST"
     --port "$PORT"
     --cache_size "$CONTEXT_SIZE"
     --grid_size "$GPU_MEM_GB")

# fp8 / nvfp4 KV lanes: hot-patch the installed engine (tools/patch_kv.py, no recompile)
case "$CACHE_QUANT" in
    fp8|nvfp4)
        "$PYTHON" tools/patch_kv.py apply || { echo "ERROR: could not enable CACHE_QUANT=$CACHE_QUANT (see above)" >&2; exit 1; }
        ;;
    none|[2-8]|[2-8],[2-8]) ;;
    *)  echo "CACHE_QUANT must be none, 2-8, k_bits,v_bits, fp8 or nvfp4 (got: $CACHE_QUANT)" >&2; exit 1 ;;
esac
if [ "$CACHE_QUANT" != "none" ]; then
    cmd+=(--cache_quant "$CACHE_QUANT")
fi
case "$DRAFT" in
    mtp)      cmd+=(--draft_model mtp) ;;
    none)     cmd+=(--draft_model none) ;;
    *)
        echo "DRAFT must be mtp or none (got: $DRAFT)" >&2
        exit 1
        ;;
esac
if [ "$CPU_CACHE_GB" != "0" ]; then
    cmd+=(--cpu_cache_size "$CPU_CACHE_GB")
fi

echo "Starting: ${cmd[*]}"

# SIMPLEX_SUPERVISED lets the built-in UI offer "switch model": the server then
# exits with 87 after writing the new settings into .env, and we start over
# from the top of this script so every setting is re-read.
export SIMPLEX_SUPERVISED=1
"${cmd[@]}"
code=$?
if [ "$code" = "87" ]; then
    echo
    echo "Switching model - reloading with the new settings from .env ..."
    echo
    exec "$0" "$@"
fi
exit $code

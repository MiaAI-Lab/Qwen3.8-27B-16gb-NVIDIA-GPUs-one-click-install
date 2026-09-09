#!/usr/bin/env bash
# Start the OpenAI-compatible exllamav3 server (tools/serve_openai.py).
# Configuration lives in .env — created from .env.example on first run.
#
# Works from the deployment kit or from the engine repo itself. First run
# builds .venv, installs torch + the engine (compiling the CUDA kernels) +
# server deps, then downloads the model weights from Hugging Face and serves.
# Later runs start the server directly.
set -euo pipefail
# This script lives in linux/; the kit is its parent.
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SELF_DIR/.."

# --- background mode ------------------------------------------------------
# On Windows the kit lives in the tray with no console; the honest equivalent
# here is simply not holding a terminal open. --background re-runs this script
# detached with its output in logs/, and prints where that log is and how to
# stop it. Everything else - first-run setup, the profile menu - happens
# exactly as it would in front of you, written to the log instead of the
# screen. --status says whether one is up; stop.sh stops it, and already
# refuses to kill anything that is not this server.
PIDFILE="logs/server.pid"
_port_from_env() {
    local p=""
    [ -f .env ] && p="$(sed -n 's/^PORT=\([0-9]\{1,\}\).*/\1/p' .env | head -1)"
    echo "${p:-8888}"
}
_running_pid() {
    [ -f "$PIDFILE" ] || return 1
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    echo "$pid"
}

RUN_BG=0
_args=()
for _a in "$@"; do
    case "$_a" in
        --background|-b) RUN_BG=1 ;;
        --status)
            _p="$(_port_from_env)"
            if pid="$(_running_pid)"; then
                echo "Running in the background (pid $pid) - http://127.0.0.1:$_p/"
                echo "Log:  $(ls -t logs/simplex-*.log 2>/dev/null | head -1)"
                exit 0
            fi
            if command -v curl >/dev/null 2>&1 \
               && curl -fsS --max-time 2 "http://127.0.0.1:$_p/health" >/dev/null 2>&1; then
                echo "Something is serving on port $_p, but not started by --background."
                exit 0
            fi
            echo "Not running."
            exit 1 ;;
        *) _args+=("$_a") ;;
    esac
done
set -- ${_args[@]+"${_args[@]}"}

# --- setup, or start ------------------------------------------------------
# setup.sh installs and downloads; start.sh starts one of the models that are
# already here. They share this file because the machinery below - the venv,
# the engine guard, the .env reader - belongs to both; the mode only decides
# where it stops. --no-pick is internal: a model switch re-execs this script
# and must not be asked all over again which model it meant.
MODE=start
PICK=1
_mode_args=()
for _a in "$@"; do
    case "$_a" in
        setup|--setup|install|--install|profile|--profile) MODE=setup ;;
        --no-pick) PICK=0 ;;
        # The harness on or off for this run only. SIMPLEX_UI rather than UI,
        # because .env is read after the environment and UI= in it would win.
        # exported: --background re-runs this script through nohup without
        # the flag (it has already been stripped here), so the child has to
        # inherit the answer rather than be told again.
        --harness)    export SIMPLEX_UI=browser ;;
        --no-harness) export SIMPLEX_UI=no ;;
        *) _mode_args+=("$_a") ;;
    esac
done
set -- ${_mode_args[@]+"${_mode_args[@]}"}

_ask_yes() {   # _ask_yes "question"  -> 0 for yes. No terminal means yes.
    [ -t 0 ] || return 0
    printf '  ? %s  [Enter = yes, n = no]: ' "$1"
    local ans=""
    read -r ans || ans=""
    case "$ans" in [nN]*) return 1 ;; *) return 0 ;; esac
}

if [ "$RUN_BG" = "1" ]; then
    if pid="$(_running_pid)"; then
        echo "Already running in the background (pid $pid)."
        echo "Stop it with ./stop.sh, or watch it with:  tail -f $(ls -t logs/simplex-*.log 2>/dev/null | head -1)"
        exit 0
    fi
    mkdir -p logs
    LOG="logs/simplex-$(date +%Y-%m-%d_%H%M%S).log"
    # $0 as given may be a bare name ("bash start.sh"), and the child is exec'd
    # without a PATH lookup - so it is resolved against this script's own
    # directory, which is linux/, not the kit root this script cd'd into. nohup, not setsid: setsid forks, which would
    # put its own pid in the file instead of the server's, and nohup is what
    # actually keeps this alive when the terminal closes.
    SELF="$SELF_DIR/$(basename "$0")"
    # through the same shell rather than executing the file: a checkout that
    # arrived without its exec bit (a zip, a copy off Windows) would otherwise
    # fail here with "Permission denied" and nowhere obvious to look
    nohup "${BASH:-bash}" "$SELF" ${1+"$@"} >"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    port="$(_port_from_env)"
    echo "Starting in the background (pid $(cat "$PIDFILE"))."
    echo "  log:   $LOG"
    echo "  stop:  ./stop.sh"
    echo "  check: ./start.sh --status"
    # A first run downloads weights and takes minutes, so this does not wait
    # for ready - only long enough to catch a start that dies immediately.
    sleep 3
    if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo
        echo "It stopped straight away. The reason is at the end of the log:" >&2
        tail -n 20 "$LOG" >&2
        rm -f "$PIDFILE"
        exit 1
    fi
    echo
    echo "It is starting. When the model is loaded it answers on http://127.0.0.1:$port/"
    exit 0
fi

if [ ! -f tools/serve_openai.py ]; then
    echo "start.sh must run from the deployment kit or the engine repo" >&2
    echo "(tools/serve_openai.py not found next to it)." >&2
    exit 1
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "No .env found - created one from .env.example."
    if [ "$MODE" = "setup" ]; then
        # Carry on. Stopping here sent a first-time user who had just run
        # ./setup.sh away to ./start.sh, which is the other door and does not
        # install anything; win_start.py has always created the file and
        # continued, so the two systems disagreed on the very first command.
        echo "Continuing with the defaults in it - setup asks the rest."
    else
        echo "Run ./setup.sh to choose a model and install, or edit .env and"
        echo "run ./start.sh again."
        exit 1
    fi
fi
# shellcheck disable=SC1091
# GPU-aware profile (tools/profiles.py): detect VRAM, choose quant / context /
# KV cache, write the choice into .env. That is setup's question. A start only
# asks it when .env has no answer in it at all, because then there is nothing
# to start and sending the user off to another script would be a dead end.
_pyprof="$(command -v python3 || command -v python || true)"
if [ -n "$_pyprof" ]; then
    if [ "$MODE" = "setup" ]; then
        "$_pyprof" tools/profiles.py --force || exit 1
    elif ! grep -q '^PROFILE=' .env || grep -qi '^PROFILE=ask' .env; then
        echo "No model has been chosen yet - this kit has not been set up."
        _ask_yes "Set it up now?" || {
            echo "  Run ./setup.sh when you want to install and download a model."
            exit 3
        }
        "$_pyprof" tools/profiles.py || exit 1
    fi
fi

# .env is data, not shell. `source` runs it, and this file is written by
# start.bat and tools/profiles.py as much as by hand, so two ordinary lines in
# it are commands to bash: a value with a space (PROFILE_GPU=NVIDIA GeForce RTX
# 5090 tries to run GeForce) and the carriage return every line carries when
# the file was last written on Windows ($'\r': command not found). Read it as
# the key=value list it is instead - same rules as the Python side, which has
# always parsed it rather than executing it.
_load_env() {
    local line key value
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            [A-Za-z_]*) ;;
            *) continue ;;
        esac
        case "$key" in *[!A-Za-z0-9_]*) continue ;; esac
        # a trailing comment, the way the Python parser reads one
        case "$value" in
            \'*\'|\"*\") value="${value:1:${#value}-2}" ;;
            *" #"*) value="${value%% #*}" ;;
        esac
        # trailing whitespace only; a value's own spaces are its own business
        while [ "${value% }" != "$value" ] || [ "${value%$'\t'}" != "$value" ]; do
            value="${value% }"; value="${value%$'\t'}"
        done
        printf -v "$key" '%s' "$value"
    done < .env
}
_load_env

# .env is sourced as shell vars; the model-download subprocess needs the HF
# token in its environment, so export it if set.
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi

# --- bootstrap: build the venv + install the engine on first run ----------
# Re-enters if the venv is missing OR the install is incomplete (e.g. a
# Ctrl-C during the first run left a half-installed venv) — pip is idempotent.
if [ ! -x .venv/bin/python ] \
   || ! .venv/bin/python -c "import torch, exllamav3, aiohttp, huggingface_hub" 2>/dev/null; then
    if [ "$MODE" != "setup" ]; then
        echo "This kit is not ready to start: the Python environment is missing or incomplete."
        _ask_yes "Install it now?" || {
            echo "  Run ./setup.sh when you want to install it."
            exit 3
        }
    fi
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
    # nvidia-* runtime wheels resolve too (the local-version wheel wins).
    #
    # cu128 rather than the newest line, because the engine's own release
    # builds for cu128 and cu132 only - torch from cu130 means no prebuilt
    # engine wheel exists and every user compiles. cu128 covers Blackwell and
    # needs driver 570+. aarch64 (GB10) has no prebuilt engine wheel on any
    # line, so it keeps cu130. TORCH_INDEX_URL in .env overrides either.
    if [ "$(uname -m)" = "aarch64" ]; then
        _default_torch_index="https://download.pytorch.org/whl/cu130"
    else
        _default_torch_index="https://download.pytorch.org/whl/cu128"
    fi
    _torch_index="${TORCH_INDEX_URL:-$_default_torch_index}"
    # Which torch, not only which index. The engine publishes wheels for
    # certain (CUDA line x torch x Python) combinations only, and the index
    # ceiling and the engine's ceiling do not have to agree: cu128 serves
    # torch 2.11 to every Python, while the engine builds 2.11 for 3.12+ only.
    # Uncapped, a 3.10 user gets a torch no wheel matches and compiles for
    # twenty minutes. tools/wheels.py owns that table and answers from it;
    # it prints a plain "torch" when capping would buy nothing.
    _torch_req="$(.venv/bin/python tools/wheels.py --torch-req --cuda "$_torch_index" 2>/dev/null || echo torch)"
    [ -n "$_torch_req" ] || _torch_req=torch
    # Output NOT hidden: pip's own download progress bars show here.
    _step "3/5 PyTorch (~2–3 GB download the first time)" \
        .venv/bin/pip install "$_torch_req" \
            --extra-index-url "$_torch_index"
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
    # The engine's own release publishes a wheel per (CUDA line x torch x
    # Python). When one fits this venv, installing it is a ~100 MB download
    # instead of a 5-20 minute compile that needs the CUDA Toolkit.
    _engine_wheel=""
    if [ "$_engine_src" != "." ] && [ -z "${EXL3_REPO:-}" ]; then
        _engine_wheel="$(.venv/bin/python tools/wheels.py --url --python .venv/bin/python 2>/dev/null || true)"
    fi
    if [ -n "$_engine_wheel" ]; then
        _quiet_step "4/5 exllamav3 engine — prebuilt wheel (no compiler needed)" \
            .venv/bin/pip install --only-binary :all: --no-build-isolation "$_engine_wheel" \
            || _engine_wheel=""
    fi
    if [ -z "$_engine_wheel" ]; then
        _quiet_step "4/5 ${_engine_note} (5–20 min depending on machine)" \
            .venv/bin/pip install --no-build-isolation \
                "${_engine_src}"
    fi
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

# --- which model? ---------------------------------------------------------
# The one question a start has that setup does not: several sizes can be on
# the disk, and only one of them is being asked for now. Enter is whatever ran
# last; the answer is written into .env, so everything below reads it exactly
# the way it always has.
if [ "$MODE" = "start" ] && [ "$PICK" = "1" ]; then
    _pick=0
    "$PYTHON" tools/profiles.py --pick || _pick=$?
    case "$_pick" in
        0) ;;
        2)  echo "Nothing is fully downloaded yet."
            _ask_yes "Choose a model and download it now?" || {
                echo "  Run ./setup.sh when you want to." ; exit 3
            }
            "$PYTHON" tools/profiles.py --force || exit 1 ;;
        3)  echo "Nothing started." ; exit 0 ;;
        *)  echo "Could not work out which model to start (code $_pick)." >&2 ; exit "$_pick" ;;
    esac
    _load_env
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
    # "config.json exists" used to be the test, and config.json is one of the
    # first small files a download fetches - so a run that stopped in the
    # middle of a shard was called "already present", and the server then met
    # half a model. tools/downloader.py answers the question properly (every
    # shard the index names, and no .part still waiting) and is what the
    # Windows launcher and the setup page ask as well.
    if "$PYTHON" -c "import sys; sys.path.insert(0, 'tools'); import downloader; sys.exit(0 if downloader.is_complete(sys.argv[1]) else 1)" "$dir" 2>/dev/null; then
        echo "$label: $dir already present — skipping download."
        return 0
    fi
    echo "$label: fetching from huggingface.co/$repo (resumes if it was interrupted) …"
    mkdir -p "$dir"
    if [ -n "${HF_REVISION:-}" ]; then
        "$PYTHON" tools/downloader.py "$repo" "$dir" --revision "$HF_REVISION"
    else
        "$PYTHON" tools/downloader.py "$repo" "$dir"
    fi
}

dl_model "$HF_TARGET_REPO" "$MODEL_DIR" "target model"

if [ "$MODE" = "setup" ]; then
    echo
    echo "Setup finished."
    echo "Start the model with ./start.sh - it asks which size to load when"
    echo "more than one has been downloaded."
    exit 0
fi

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

case "$CACHE_QUANT" in
    none|[2-8]|[2-8],[2-8]) ;;
    *)  echo "CACHE_QUANT must be none, 2-8 or k_bits,v_bits (got: $CACHE_QUANT)" >&2; exit 1 ;;
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

# --- the harness ----------------------------------------------------------
# The model server serves /v1 and a small page at / saying where things are.
# What you talk to is the DeepSeek Harness (tools/dsh.py), a separate process
# started once the model answers /health - by then the server can say what it
# actually loaded, and the harness is configured from that rather than from
# the hopes in this file.
#   UI=browser  start it and open it   (default)
#   UI=server   start it, open nothing
#   UI=no       do not start it; /v1 still serves every other client
UI="${UI:-browser}"
# --harness / --no-harness, and `simplex start --no-harness`, which sets this.
UI="${SIMPLEX_UI:-$UI}"
DSH_PORT="${DSH_PORT:-3080}"
case "$UI" in
    1|yes|true|on|browser) UI=browser ;;
    0|no|none|off|false)   UI=no ;;
    server) ;;
    *) echo "UI=$UI is not browser/server/no - using browser" >&2; UI=browser ;;
esac
cmd+=(--harness_port "$DSH_PORT")
[ "$UI" = "no" ] && cmd+=(--ui off)

if [ "$UI" != "no" ]; then
    _dsh_args=(--wait 3600 --port "$DSH_PORT" --base "http://127.0.0.1:$PORT/v1")
    [ "$UI" = "browser" ] && _dsh_args+=(--open)
    "$PYTHON" -u tools/dsh.py "${_dsh_args[@]}" &
    DSH_PID=$!
    # Only on the way out for good. A model switch re-execs this script, which
    # replaces the process without running traps - deliberately, because the
    # harness survives the switch and dsh.py notices the port is already held.
    trap 'kill "$DSH_PID" 2>/dev/null || true' EXIT INT TERM
fi

echo "Starting: ${cmd[*]}"

# Exit code 87 is "load another model": whatever asked for it has already
# written the new settings into .env, so we start over from the top of this
# script and every setting is re-read.
# `|| code=$?`, not a bare call: this script runs under `set -e`, which exits on
# the spot when a simple command fails - so a server returning 87 killed the
# launcher before the line below could read the code, and the model switch has
# never once reached its reload. A non-zero exit is expected here; it is data.
code=0
"${cmd[@]}" || code=$?
if [ "$code" = "87" ]; then
    echo
    echo "Switching model - reloading with the new settings from .env ..."
    echo
    exec "$0" --no-pick "$@"
fi
exit $code

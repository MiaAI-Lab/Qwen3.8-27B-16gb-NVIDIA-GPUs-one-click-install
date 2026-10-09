#!/usr/bin/env bash
# Small helpers sourced by linux/start.sh. No side effects on load.

# Total / available system RAM in GiB, independent of the user's locale.
# `free` translates its row labels ("Mem:" -> "Mem.:", "Memória:"), so the
# label is never matched; the second line is always the memory row.
_free_gib() {   # $1 = column number in `free -g` (2 total, 7 available)
    LC_ALL=C free -g | awk -v c="$1" 'NR==2 {print $c; exit}'
}

# Where the CUDA toolkit lives. Honours CUDA_HOME if it has nvcc; otherwise
# looks at nvcc on PATH and the usual install roots (Debian/Ubuntu
# /usr/local/cuda, Arch /opt/cuda). Prints the directory, or nothing.
detect_cuda_home() {
    local c nvcc
    if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
        echo "$CUDA_HOME"; return 0
    fi
    if nvcc="$(command -v nvcc 2>/dev/null)" && [ -n "$nvcc" ]; then
        c="$(cd "$(dirname "$(readlink -f "$nvcc")")/.." 2>/dev/null && pwd)" || c=""
        if [ -n "$c" ]; then echo "$c"; return 0; fi
    fi
    for c in /usr/local/cuda /opt/cuda /usr/lib/cuda; do
        if [ -x "$c/bin/nvcc" ]; then echo "$c"; return 0; fi
    done
    # versioned installs, newest last
    for c in $(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tac); do
        if [ -x "$c/bin/nvcc" ]; then echo "$c"; return 0; fi
    done
    return 1
}

# Source builds only. CUDA 12.x's nvcc (and torch's extension builder) rejects a
# host compiler of GCC 14 or newer, which rolling-release distros ship. Fail
# before the multi-GB torch download rather than after it. Prints a message and
# returns 1 when the build would fail; 0 otherwise (including when it cannot
# tell). SKIP_COMPILER_CHECK=1 turns it off.
check_host_compiler() {   # $1 = CUDA_HOME
    [ "${SKIP_COMPILER_CHECK:-0}" = 1 ] && return 0
    local cxx="${CXX:-g++}" ver major nvcc="$1/bin/nvcc" cuda_major
    command -v "$cxx" >/dev/null 2>&1 || return 0
    [ -x "$nvcc" ] || return 0
    major="$("$cxx" -dumpversion 2>/dev/null | cut -d. -f1)"
    cuda_major="$("$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]\{1,\}\)\..*/\1/p' | head -1)"
    case "$major" in ''|*[!0-9]*) return 0 ;; esac
    case "$cuda_major" in ''|*[!0-9]*) return 0 ;; esac
    if [ "$cuda_major" -le 12 ] && [ "$major" -ge 14 ]; then
        echo "ERROR: the engine has to be compiled here, but CUDA $cuda_major.x does not accept $cxx $major." >&2
        echo "       Install an older compiler (e.g. gcc-13/g++-13) and run:" >&2
        echo "         CC=gcc-13 CXX=g++-13 ./linux/start.sh" >&2
        echo "       (SKIP_COMPILER_CHECK=1 skips this check.)" >&2
        return 1
    fi
    return 0
}

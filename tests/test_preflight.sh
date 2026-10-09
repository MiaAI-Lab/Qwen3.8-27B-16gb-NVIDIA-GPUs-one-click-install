#!/usr/bin/env bash
# Tests for linux/preflight.sh. Run: bash tests/test_preflight.sh
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../linux/preflight.sh
. "$HERE/linux/preflight.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "ok   - $1"; }
bad() { echo "FAIL - $1"; fail=1; }

# --- _free_gib: a translated label must not break it ------------------------
mkdir -p "$T/bin"
cat > "$T/bin/free" <<'F'
#!/usr/bin/env bash
echo "               total        usado       livre"
echo "Mem.:             62          10          40          1          11          50"
echo "Swap:              0           0           0"
F
chmod +x "$T/bin/free"
got="$(PATH="$T/bin:$PATH" _free_gib 2)"
[ "$got" = 62 ] && ok "_free_gib reads a translated Mem row (total)" || bad "_free_gib total: '$got'"
got="$(PATH="$T/bin:$PATH" _free_gib 7)"
[ "$got" = 50 ] && ok "_free_gib reads a translated Mem row (available)" || bad "_free_gib available: '$got'"

# --- detect_cuda_home -------------------------------------------------------
mk_nvcc() { mkdir -p "$1/bin"; printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release %s, V%s.0"\n' "$2" "$2" > "$1/bin/nvcc"; chmod +x "$1/bin/nvcc"; }
mk_nvcc "$T/opt/cuda" 12.8
got="$(env -u CUDA_HOME PATH="$T/opt/cuda/bin:/usr/bin:/bin" bash -c ". '$HERE/linux/preflight.sh'; detect_cuda_home")"
[ "$got" = "$T/opt/cuda" ] && ok "detect_cuda_home follows nvcc on PATH" || bad "detect_cuda_home PATH: '$got'"
got="$(CUDA_HOME="$T/opt/cuda" PATH=/usr/bin:/bin bash -c ". '$HERE/linux/preflight.sh'; detect_cuda_home")"
[ "$got" = "$T/opt/cuda" ] && ok "detect_cuda_home keeps a valid CUDA_HOME" || bad "detect_cuda_home env: '$got'"
got="$(CUDA_HOME="$T/nope" PATH="$T/opt/cuda/bin:/usr/bin:/bin" bash -c ". '$HERE/linux/preflight.sh'; detect_cuda_home")"
[ "$got" = "$T/opt/cuda" ] && ok "detect_cuda_home ignores a CUDA_HOME without nvcc" || bad "detect_cuda_home bad env: '$got'"

# --- check_host_compiler ----------------------------------------------------
mk_gxx() { mkdir -p "$T/gcc$1"; printf '#!/usr/bin/env bash\necho %s\n' "$1" > "$T/gcc$1/g++"; chmod +x "$T/gcc$1/g++"; }
mk_gxx 16; mk_gxx 13
CXX="$T/gcc16/g++" check_host_compiler "$T/opt/cuda" 2>/dev/null && bad "GCC 16 + CUDA 12.8 must be refused" || ok "GCC 16 + CUDA 12.8 refused"
CXX="$T/gcc13/g++" check_host_compiler "$T/opt/cuda" 2>/dev/null && ok "GCC 13 + CUDA 12.8 accepted" || bad "GCC 13 + CUDA 12.8 refused"
SKIP_COMPILER_CHECK=1 CXX="$T/gcc16/g++" check_host_compiler "$T/opt/cuda" 2>/dev/null && ok "SKIP_COMPILER_CHECK=1 skips the check" || bad "skip flag ignored"
mk_nvcc "$T/cuda13" 13.0
CXX="$T/gcc16/g++" check_host_compiler "$T/cuda13" 2>/dev/null && ok "CUDA 13 is not judged by the 12.x rule" || bad "CUDA 13 wrongly refused"
check_host_compiler "$T/missing" 2>/dev/null && ok "unknown toolkit: no verdict" || bad "unknown toolkit refused"

exit $fail

#!/usr/bin/env bash
# stop.sh - stop the server started by start.sh, and the harness with it.
#
# The work is in tools/cli.py (`simplex stop`), which is the same code the
# Windows stop runs. This file was a second implementation in bash, and it had
# bugs the Python one does not: it needed `ss` from iproute2, and it sent the
# server SIGINT - which a background job's child inherits as SIG_IGN, so every
# background stop sat out ten seconds and finished with a SIGKILL.
#
#   ./stop.sh                 stop both
#   ./stop.sh --harness-only  leave the model loaded
#   ./stop.sh --server-only   leave the harness running
set -euo pipefail
# This script lives in linux/; the kit is its parent.
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SELF_DIR/.."
PY="$(command -v python3 || command -v python || true)"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi
if [ -z "$PY" ]; then
    echo "Python 3.11+ is needed and was not found on PATH." >&2
    exit 1
fi
exec "$PY" tools/cli.py stop "$@"

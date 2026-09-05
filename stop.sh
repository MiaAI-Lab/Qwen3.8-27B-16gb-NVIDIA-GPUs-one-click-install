#!/usr/bin/env bash
# stop.sh — stop the server started by start.sh (graceful, then forced)
# Reads PORT from .env (default 8888). Safe to run when nothing is up.
set -euo pipefail
cd "$(dirname "$0")"
# Explicit env var beats .env; .env beats built-in default
ENV_PORT="${PORT:-}"
[ -f .env ] && source .env
PORT="${ENV_PORT:-${PORT:-8888}}"

pid="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)"

if [ -z "$pid" ]; then
    echo "No server listening on port $PORT."
    exit 0
fi

# Safety: only kill if the listener is actually our server
cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
if ! grep -q serve_openai <<< "$cmdline"; then
    echo "Port $PORT is held by another process (pid $pid): ${cmdline:0:70}"
    echo "Refusing to kill it."
    exit 1
fi

echo "Stopping server (pid $pid) on port $PORT..."
kill -INT "$pid" 2>/dev/null || true
for _ in $(seq 1 20); do
    if ! ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q .; then
        echo "Stopped."
        exit 0
    fi
    sleep 0.5
done

echo "Still running after 10s, sending SIGKILL..."
kill -KILL "$pid" 2>/dev/null || true
sleep 1
ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q . && echo "ERROR: port $PORT still open" && exit 1
echo "Stopped."

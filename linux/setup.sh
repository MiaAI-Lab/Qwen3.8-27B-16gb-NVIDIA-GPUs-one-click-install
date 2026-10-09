#!/usr/bin/env bash
# Setup only: pick a model size for this graphics card, build the kit's own
# .venv, install the engine, download the weights - then stop.
#
# Starting a model is ./start.sh's job. Keeping the two apart is what lets you
# download a second size without also loading one, and load one without being
# asked anything about downloads.
set -euo pipefail
# This script lives in linux/; the kit is its parent.
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SELF_DIR/.."
exec "${BASH:-bash}" "$SELF_DIR/start.sh" setup "$@"

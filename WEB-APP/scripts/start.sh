#!/bin/bash
set -e
cd "$(dirname "$0")/.."
VENV_DIR="$(python3 scripts/venv_path.py)"
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "[错误] 请先运行 ./scripts/install.sh。" >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$VENV_DIR/bin/python" -B run.py "$@"

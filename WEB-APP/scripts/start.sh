#!/bin/bash
set -e
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
    echo "[错误] 请先运行 ./scripts/install.sh。" >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -B run.py "$@"

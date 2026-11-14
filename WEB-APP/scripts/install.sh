#!/bin/bash
set -e
cd "$(dirname "$0")/.."

command -v python3 >/dev/null 2>&1 || { echo "[错误] 请先安装 Python 3.10+。" >&2; exit 1; }
python3 -c "import sys; assert sys.version_info >= (3, 10), '需要 Python 3.10+'"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
echo "安装完成：./scripts/start.sh"
echo "可选 Excel：.venv/bin/python -m pip install 'openpyxl>=3.1.0'"

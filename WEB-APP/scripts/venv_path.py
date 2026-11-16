"""Resolve the per-checkout virtual environment outside the project."""

import hashlib
import os
from pathlib import Path


def venv_path():
    override = os.environ.get("LMSC_VENV_DIR")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError("LMSC_VENV_DIR must be an absolute path")
        return path
    root = Path(__file__).resolve().parent.parent
    identity = os.path.normcase(str(root))
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) if os.name == "nt" else Path.home() / ".local" / "share"
    return base / "LMSC" / "venvs" / suffix


if __name__ == "__main__":
    print(venv_path())

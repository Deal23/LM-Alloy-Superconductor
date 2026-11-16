"""Register a per-user Windows launcher without replacing other commands."""

import ctypes
import os
from pathlib import Path
import shutil
import sys
import winreg


def main():
    root = Path(__file__).resolve().parent.parent
    launcher_dir = Path(os.environ["LOCALAPPDATA"]) / "LMSC" / "bin"
    launcher = launcher_dir / "lmsc.cmd"
    marker = "rem LMSC web app launcher"

    existing = shutil.which("lmsc")
    if existing and Path(existing).resolve() != launcher.resolve():
        print(f"lmsc already exists at {existing}; registration skipped.")
        return 1
    if launcher.exists() and marker not in launcher.read_text(encoding="mbcs"):
        print(f"Existing launcher at {launcher} is not managed by LMSC; registration skipped.")
        return 1

    launcher_dir.mkdir(parents=True, exist_ok=True)
    launcher.write_text(
        f'@echo off\n{marker}\n"{root / ".venv" / "Scripts" / "python.exe"}" '
        f'-B "{root / "run.py"}" %*\n',
        encoding="mbcs",
    )

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                        winreg.KEY_READ | winreg.KEY_SET_VALUE) as key:
        try:
            path, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            path, kind = "", winreg.REG_EXPAND_SZ
        entries = [entry.strip().strip('"') for entry in path.split(";") if entry.strip()]
        if not any(os.path.normcase(os.path.normpath(os.path.expandvars(entry))) ==
                   os.path.normcase(os.path.normpath(str(launcher_dir))) for entry in entries):
            updated = path.rstrip(";") + (";" if path else "") + str(launcher_dir)
            winreg.SetValueEx(key, "Path", 0, kind, updated)
            ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 2, 5000, None)
    print(f"Registered lmsc: {launcher}. Open a new terminal to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

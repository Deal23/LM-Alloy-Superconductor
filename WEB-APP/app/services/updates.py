"""Manual, local-only updates from a pinned default-branch commit."""
import ast
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import threading
from urllib.request import Request, urlopen
import zipfile

from app import __version__

REPOSITORY = "Deal23/LM-Alloy-Superconductor"
API = f"https://api.github.com/repos/{REPOSITORY}"
MAX_DOWNLOAD = 150 * 1024 * 1024
MAX_EXPANDED = 600 * 1024 * 1024
_lock = threading.Lock()


def _version(value):
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", value or "")
    if not match:
        raise ValueError("版本号必须是 v主版本.次版本.修订号")
    return tuple(map(int, match.groups()))


def _get(url, limit):
    request = Request(url, headers={"User-Agent": "LMSC-Web-App", "Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=20) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("下载内容超过大小限制")
    return data


def _module_version(content):
    module = ast.parse(content)
    versions = [node.value.value for node in module.body
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
    if len(versions) != 1 or not isinstance(versions[0], str):
        raise ValueError("无法读取 Web 应用版本")
    _version(versions[0])
    return versions[0]


def latest_release():
    repository = json.loads(_get(API, 1024 * 1024))
    branch = repository["default_branch"]
    from urllib.parse import quote
    commit = json.loads(_get(f"{API}/commits/{quote(branch, safe='')}", 1024 * 1024))["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("远端提交标识无效")
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/WEB-APP/app/__init__.py"
    version = _module_version(_get(url, 1024 * 1024).decode("utf-8"))
    return {"current": __version__, "latest": version, "newer": _version(version) > _version(__version__),
            "ready": True, "notes": "来自仓库默认分支。", "url": f"https://github.com/{REPOSITORY}/commit/{commit}",
            "commit": commit}


def _allowed(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(p in (".", "..") for p in path.parts):
        return False
    if len(path.parts) == 1:
        return name in {"run.py", "requirements.txt", "README.md", "LICENSE", "CITATION.cff"}
    if path.parts[0] not in {"app", "ml", "utils", "scripts"}:
        return False
    return not (path.parts[:2] == ("ml", "models") or any(p.startswith(".") or p == "__pycache__" for p in path.parts))


def apply_update(root, expected):
    """Download a checked release and replace program files, retaining a backup."""
    if not _lock.acquire(blocking=False):
        raise ValueError("更新正在进行")
    try:
        release = latest_release()
        if not release["newer"] or not release["ready"] or release["latest"] != expected:
            raise ValueError("版本已变化或没有可用更新包，请重新检查")
        archive = _get(f"https://codeload.github.com/{REPOSITORY}/zip/{release['commit']}", MAX_DOWNLOAD)

        root = Path(root).resolve()
        with tempfile.TemporaryDirectory(prefix="lmsc-update-") as tmp:
            import io
            staged = Path(tmp) / "staged"
            staged.mkdir()
            with zipfile.ZipFile(io.BytesIO(archive)) as z:
                infos = z.infolist()
                if not infos or len(infos) > 20000 or sum(i.file_size for i in infos) > MAX_EXPANDED:
                    raise ValueError("更新包内容不合法或过大")
                names = set()
                for info in infos:
                    parts = PurePosixPath(info.filename).parts
                    if "\\" in info.filename or info.filename.startswith("/") or any(p in (".", "..") for p in parts):
                        raise ValueError("更新包含不允许的文件")
                    if len(parts) < 3 or parts[1] != "WEB-APP" or info.is_dir():
                        continue
                    name = "/".join(parts[2:])
                    if not _allowed(name):
                        continue
                    if name in names or stat.S_ISLNK(info.external_attr >> 16):
                        raise ValueError("更新包含不允许的文件")
                    names.add(name)
                    dest = root.joinpath(*PurePosixPath(name).parts)
                    if not dest.resolve().is_relative_to(root) or any(p.is_symlink() for p in (dest, *dest.parents) if p != root):
                        raise ValueError("更新路径包含符号链接")
                    target = staged.joinpath(*PurePosixPath(name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                if not {"app/__init__.py", "run.py", "requirements.txt"}.issubset(names):
                    raise ValueError("更新包不完整")
                if _module_version((staged / "app/__init__.py").read_text(encoding="utf-8")) != release["latest"]:
                    raise ValueError("更新包版本与发布版本不一致")
            backup_root = root / ".update-backups"
            if backup_root.is_symlink():
                raise ValueError("备份目录不能是符号链接")
            backup_root.mkdir(exist_ok=True)
            backup = Path(tempfile.mkdtemp(prefix="update-", dir=backup_root))
            changed = []
            try:
                for name in sorted(names):
                    dest = root.joinpath(*PurePosixPath(name).parts)
                    old = backup.joinpath(*PurePosixPath(name).parts)
                    existed = dest.exists()
                    if existed:
                        old.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(dest, old)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    changed.append((dest, old, existed))
                    shutil.copyfile(staged.joinpath(*PurePosixPath(name).parts), dest)
                    if not existed and name.startswith("scripts/") and name.endswith(".sh"):
                        dest.chmod(0o755)
            except Exception:
                for dest, old, existed in reversed(changed):
                    if existed:
                        shutil.copy2(old, dest)
                    elif dest.exists():
                        dest.unlink()
                raise
        return {"version": release["latest"], "backup": backup.relative_to(root).as_posix()}
    finally:
        _lock.release()


def is_local(address):
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False

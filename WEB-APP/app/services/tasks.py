"""
Lightweight background task runner.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import uuid
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TaskService:
    """Run long ML scripts in background threads."""

    _status: Dict[str, str] = {}
    _processes: Dict[str, subprocess.Popen] = {}
    _progress: Dict[str, Dict[str, Any]] = {}
    _lock = threading.RLock()

    SCRIPT_MAP = {
        "alloy_predict": "ml/screening/alloy_predict.py",
    }

    @classmethod
    def submit(cls, project_root: str, task_name: str, params: Optional[Dict] = None) -> Optional[str]:
        if task_name not in cls.SCRIPT_MAP:
            return None

        script_path = os.path.join(project_root, cls.SCRIPT_MAP[task_name])
        if not os.path.exists(script_path):
            return None

        task_id = str(uuid.uuid4())
        with cls._lock:
            cls._status[task_id] = "queued"
            cls._progress[task_id] = {"started_at": time.time(), "completed": 0, "total": None, "eta_seconds": None}

        thread = threading.Thread(
            target=cls._run,
            args=(project_root, script_path, task_id, params or {}),
            daemon=True,
        )
        thread.start()
        return task_id

    @classmethod
    def _run(cls, project_root: str, script_path: str, task_id: str, params: Dict[str, Any]):
        with cls._lock:
            cls._status[task_id] = "running"

        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        env["LMSC_TASK_PARAMS"] = json.dumps(params, ensure_ascii=False)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        os.makedirs(os.path.join(project_root, "results", "predictions"), exist_ok=True)

        try:
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=project_root,
                env=env,
            )
            with cls._lock:
                cls._processes[task_id] = proc

            lines = []
            def read_output():
                for line in proc.stdout:
                    if line.startswith("LMSC_PROGRESS "):
                        try:
                            event = json.loads(line[len("LMSC_PROGRESS "):])
                            done, total = int(event["completed"]), int(event["total"])
                            with cls._lock:
                                progress = cls._progress[task_id]
                                elapsed = time.time() - progress["started_at"]
                                # Resumed rows are not newly measured work.
                                if "baseline" not in progress:
                                    progress["baseline"] = max(0, done - 1)
                                measured = done - progress["baseline"]
                                eta = elapsed / measured * (total - done) if measured > 0 else None
                                progress.update(completed=done, total=total, eta_seconds=round(max(0, eta)) if eta is not None else None)
                        except (ValueError, KeyError, TypeError):
                            logger.warning("Invalid screening progress event")
                    else:
                        lines.append(line)
                        if len(lines) > 200:
                            del lines[:-200]
            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            proc.wait(timeout=3600)
            reader.join(timeout=2)
            stdout = "".join(lines)
            with cls._lock:
                cls._processes.pop(task_id, None)
                if cls._status.get(task_id) == "stopped by user":
                    pass
                elif proc.returncode == 0:
                    cls._status[task_id] = f"completed\n\nOutput:\n{stdout}"
                else:
                    cls._status[task_id] = f"failed: return code {proc.returncode}\n\n{stdout}"
        except subprocess.TimeoutExpired:
            with cls._lock:
                proc = cls._processes.pop(task_id, None)
            if proc:
                proc.kill()
            cls._status[task_id] = "timeout: exceeded 1 hour"
        except Exception as e:
            logger.exception("Task runner failed")
            with cls._lock:
                cls._status[task_id] = f"failed: {e}"

    @classmethod
    def status(cls, task_id: str) -> Dict[str, Any]:
        with cls._lock:
            raw = cls._status.get(task_id, "unknown")
            progress = dict(cls._progress.get(task_id, {}))
        return {"status": raw, "progress": progress}

    @classmethod
    def stop(cls, task_id: str) -> Dict[str, Any]:
        with cls._lock:
            proc = cls._processes.get(task_id)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            with cls._lock:
                cls._processes.pop(task_id, None)
                cls._status[task_id] = "stopped by user"
            return {"success": True, "message": "Task stopped"}
        return {"success": False, "message": "Task not running"}

    @classmethod
    def list_scripts(cls) -> Dict[str, str]:
        return dict(cls.SCRIPT_MAP)

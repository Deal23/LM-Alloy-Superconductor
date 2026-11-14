"""Local resource information; estimates are intentionally conservative."""
import os
import subprocess
import sys


def system_resources():
    cpus = os.cpu_count() or 1
    memory = None
    if sys.platform == "darwin":
        try:
            memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    elif hasattr(os, "sysconf"):
        try:
            memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            pass
    return {"logical_cpus": cpus, "memory_bytes": memory}

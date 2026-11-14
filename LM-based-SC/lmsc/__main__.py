"""Run the workflow with python -m lmsc <command>."""
import argparse
import runpy
import subprocess
import sys

from .paths import ROOT

COMMANDS = {
    "build": "dataset",
    "train": "train",
    "predict": "discovery.predict",
    "anomaly": "analysis.anomaly",
    "scan": "discovery.scan",
    "elements": "analysis.element_frequency",
    "space": "analysis.element_space",
    "screening": "analysis.alloy_screening",
    "plot-training": "analysis.plot_training",
    "verify": "verify",
}


def main():
    parser = argparse.ArgumentParser(description="Tc prediction for alloys and metal–nonmetal compounds (no H/O)")
    parser.add_argument("command", choices=[*COMMANDS, "test"])
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        args, extra = parser.parse_known_args()
        if args.command == "test":
            return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), *extra])
    command = sys.argv.pop(1)
    runpy.run_module(f"lmsc.{COMMANDS[command]}", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

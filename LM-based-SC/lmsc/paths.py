"""Repository paths shared by the research workflow."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASET = DATA / "superconductors.csv"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DATA_RESULTS = RESULTS / "data"
TRAINING = RESULTS / "training"
PREDICTIONS = RESULTS / "predictions"


def public_path(path):
    """Use project-relative paths in saved artifacts, hiding external directories."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name

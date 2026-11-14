"""Shared paths and the paper's unnormalised element-ratio representation."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "data"
FIGURES = ROOT / "paper/figures/generated"
MODELS = ROOT / "data/models"


def parse_formula(formula):
    text = str(formula).strip()
    tokens = list(re.finditer(r"([A-Z][a-z]*)(\d*\.?\d*)", text))
    if not tokens or "".join(t.group() for t in tokens) != text:
        raise ValueError(f"Unsupported formula: {formula}")
    values = {}
    for token in tokens:
        symbol, amount = token.groups()
        amount = float(amount) if amount else 1.0
        if amount > 0:
            values[symbol] = values.get(symbol, 0.0) + amount
    if not values:
        raise ValueError(f"Empty composition: {formula}")
    return values


def features(formulas, columns=None):
    frame = pd.DataFrame([parse_formula(x) for x in formulas]).fillna(0.0)
    if columns is None:
        columns = sorted(frame.columns)
    unknown = sorted(set(frame.columns) - set(columns))
    if unknown:
        raise ValueError(f"Elements absent from model: {unknown}")
    return frame.reindex(columns=columns, fill_value=0.0)


def read_training(path):
    frame = pd.read_csv(path)
    frame["Tc"] = pd.to_numeric(frame["Tc"], errors="raise")
    if not np.isfinite(frame["Tc"]).all():
        raise ValueError("Non-finite Tc")
    return frame, features(frame["formula"]), frame["Tc"]


def evaluate(y, predicted):
    from sklearn.metrics import r2_score

    y, predicted = np.asarray(y), np.asarray(predicted)
    error = predicted - y
    denominator = np.abs(y) + np.abs(predicted)
    mask = y > 5
    return {
        "rmse_K": float(np.sqrt(np.mean(error ** 2))),
        "mae_K": float(np.mean(np.abs(error))),
        "r2": float(r2_score(y, predicted)),
        "accuracy_5K_pct": float(100 * np.mean(np.abs(error) < 5)),
        "mare_Tc_gt_5": float(np.mean(np.abs(error[mask] / y[mask]))) if mask.any() else None,
        "smape_pct": float(100 * np.mean(np.divide(2 * np.abs(error), denominator,
                                                   out=np.zeros_like(error, dtype=float),
                                                   where=denominator != 0))),
    }


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_model(name, model_dir=MODELS):
    import joblib

    model_dir = Path(model_dir)
    model = joblib.load(model_dir / f"{name}.joblib")
    json_path = model_dir / f"{name}_features.json"
    if json_path.exists():
        columns = json.loads(json_path.read_text())
    else:
        columns = joblib.load(model_dir / f"{name}_feature.joblib")
    return model, list(columns)

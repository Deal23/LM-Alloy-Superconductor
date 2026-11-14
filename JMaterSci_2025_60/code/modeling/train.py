"""Train ET/RF or compare eight regressors with explicit evaluation modes."""
import argparse
import hashlib
import time

import joblib
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, MODELS, TABLES, evaluate, read_training, write_csv, write_json


def build_model(name, seed=42, jobs=-1, paper_params=True):
    if name in ("ET", "RF"):
        cls = ExtraTreesRegressor if name == "ET" else RandomForestRegressor
        params = dict(n_estimators=300, criterion="squared_error",
                      min_samples_split=5 if name == "ET" else 2) if paper_params else {}
        return cls(random_state=seed, n_jobs=jobs, **params)
    factories = {"DT": DecisionTreeRegressor, "GB": GradientBoostingRegressor,
                 "HistGB": HistGradientBoostingRegressor}
    if name in factories:
        return factories[name](random_state=seed)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=seed, verbose=False, thread_count=jobs, allow_writing_files=False)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=jobs)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=jobs, verbosity=-1)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=str(DATA / "processed/mdr_clean.csv"))
    parser.add_argument("--models", nargs="+", default=["ET", "RF"],
                        choices=["ET", "RF", "DT", "GB", "HistGB", "CatBoost", "XGBoost", "LightGBM"])
    parser.add_argument("--compare", action="store_true", help="Use model defaults rather than selected paper parameters")
    parser.add_argument("--evaluation", choices=["holdout", "resubstitution"], default="holdout")
    parser.add_argument("--legacy-skip-first", action="store_true", help="Reproduce comparison cell's first-row omission")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--output", type=str, default=str(TABLES / "generated/training"))
    parser.add_argument("--model-dir", type=str, default=str(MODELS))
    args = parser.parse_args()
    from pathlib import Path
    output, model_dir = Path(args.output), Path(args.model_dir)
    frame, X, y = read_training(args.data)
    if args.legacy_skip_first:
        frame, X, y = frame.iloc[1:], X.iloc[1:], y.iloc[1:]
    if args.evaluation == "holdout":
        train_idx, test_idx = train_test_split(range(len(X)), test_size=0.1, random_state=args.seed)
    else:
        train_idx = test_idx = list(range(len(X)))
    records = []
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        model = build_model(name, args.seed, args.jobs, not args.compare)
        start = time.perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        scores = evaluate(y.iloc[test_idx], pred)
        records.append({"model": name, "evaluation": args.evaluation,
                        "fit_seconds": time.perf_counter() - start, **scores})
        write_csv(pd.DataFrame({"row_index": frame.index[test_idx],
                                "formula": frame.iloc[test_idx]["formula"].values,
                                "Tc": y.iloc[test_idx].values, "Predicted": pred}), output / f"{name}_predictions.csv")
        joblib.dump(model, model_dir / f"{name}.joblib", compress=3)
        write_json(list(X.columns), model_dir / f"{name}_features.json")
        write_json({"data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    "data": str(Path(args.data).name), "n_train": len(train_idx), "n_test": len(test_idx),
                    "evaluation": args.evaluation, "seed": args.seed, "parameters": model.get_params(),
                    "raw_ratios": True, **scores}, model_dir / f"{name}_metrics.json")
        print(name, scores)
    write_csv(pd.DataFrame(records), output / "model_comparison.csv")


if __name__ == "__main__":
    main()

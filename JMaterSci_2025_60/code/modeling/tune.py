"""ET/RF grid with 135 unique combinations; holdout reporting is opt-in."""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import ParameterGrid, cross_val_score, train_test_split

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, TABLES, evaluate, read_training, write_csv

# The paper's 270 entries included both squared_error and friedman_mse.
# Current scikit-learn treats them equivalently for these regressors; evaluate once.
GRID = {"n_estimators": [100, 200, 300, 400, 500], "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA / "processed/mdr_clean.csv"))
    p.add_argument("--model", choices=["ET", "RF"], default="ET")
    p.add_argument("--limit", type=int, help="Smoke-test only: limit grid entries")
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--legacy-skip-first", action="store_true")
    p.add_argument("--report-holdout", action="store_true", help="Historical diagnostic; do not select parameters using these scores")
    p.add_argument("--output", default=str(TABLES / "generated/tuning"))
    args = p.parse_args()
    from pathlib import Path
    _, X, y = read_training(args.data)
    if args.legacy_skip_first:
        X, y = X.iloc[1:], y.iloc[1:]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=42)
    cls = ExtraTreesRegressor if args.model == "ET" else RandomForestRegressor
    grid = list(ParameterGrid(GRID))
    if args.limit is not None:
        if args.limit < 1:
            p.error("--limit must be positive")
        grid = grid[:args.limit]
    rows = []
    for index, params in enumerate(grid):
        start = time.perf_counter()
        model = cls(random_state=42, n_jobs=args.jobs, **params)
        scores = cross_val_score(model, Xtr, ytr, cv=5, scoring="neg_mean_squared_error", n_jobs=1)
        row = {"trial": index, "params": json.dumps(params), "cv_neg_mse": float(scores.mean()),
               "cv_rmse_K": float(np.sqrt(-scores).mean())}
        if args.report_holdout:
            model.fit(Xtr, ytr)
            row.update({f"holdout_{k}": v for k, v in evaluate(yte, model.predict(Xte)).items()})
        row["fit_seconds"] = time.perf_counter() - start
        rows.append(row)
        write_csv(pd.DataFrame(rows), Path(args.output) / f"tuning_{args.model}.csv")
        print(f"{index + 1}/{len(grid)} CV RMSE {row['cv_rmse_K']:.4f} K", flush=True)


if __name__ == "__main__":
    main()

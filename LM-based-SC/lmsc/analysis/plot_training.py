"""Regenerate all training-result figures from saved tables and the current model."""
import json

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .. import train


def main():
    metrics = json.loads((train.TABLE_DIR / "metrics.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(train.TABLE_DIR / "model_comparison.csv")
    fold_table = pd.read_csv(train.TABLE_DIR / "model_comparison_folds.csv")
    tuning_results = pd.read_csv(train.TABLE_DIR / "tuning_results.csv")
    repeated_summary = pd.read_csv(train.TABLE_DIR / "repeated_cv_summary.csv")

    _, X, y, groups, _ = train.load_data()
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=metrics["holdout_size"],
        random_state=metrics["random_state"],
    )
    _, test_idx = next(splitter.split(X, y, groups=groups))
    model = joblib.load(train.MODEL_DIR / "model.joblib")
    pred = model.predict(X.iloc[test_idx])
    holdout_metrics = train.evaluate(y.iloc[test_idx], pred)

    train.save_plots(
        comparison,
        fold_table,
        tuning_results,
        repeated_summary,
        y.iloc[test_idx],
        pred,
        holdout_metrics,
    )
    print(f"saved -> {train.FIG_DIR.relative_to(train.ROOT)}/model_*.png")


if __name__ == "__main__":
    main()

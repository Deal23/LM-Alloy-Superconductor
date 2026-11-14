"""Historical augmentation experiments; saved paper tables remain authoritative."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, TABLES, features, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(DATA / "processed/mdr_clean.csv"))
    p.add_argument("--method", choices=["duplicate", "smote", "legacy-smote"], default="duplicate")
    p.add_argument("--factor", type=int, choices=[2, 5], default=2)
    p.add_argument("--output", type=Path, default=TABLES / "generated/augmented.csv")
    args = p.parse_args()
    frame = pd.read_csv(args.input)
    if args.method == "duplicate":
        zero = frame[frame.Tc == 0]
        result = pd.concat([frame] + [zero] * (args.factor - 1), ignore_index=True)
    else:
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import StandardScaler

        X = features(frame.formula)
        scaler = StandardScaler()
        labels = np.ceil(frame.Tc.to_numpy())
        target = int((labels == 0).sum()) * args.factor
        values, labels = SMOTE(sampling_strategy={0: target}, random_state=42).fit_resample(scaler.fit_transform(X), labels)
        if args.method == "smote":
            # Unlike the original notebook, undo scaling before reconstructing formulas.
            synthetic = np.maximum(scaler.inverse_transform(values[len(frame):]), 0)
            formulas = ["".join(f"{e}{v:.3f}" for e, v in zip(X.columns, row) if v > 0)
                        for row in synthetic]
            result = pd.concat([frame.assign(type="original"),
                                pd.DataFrame({"formula": formulas, "Tc": 0, "type": "synthetic"})], ignore_index=True)
        else:
            # Explicitly reproduce the notebook defect for inspection, never silently.
            formulas = ["".join(f"{e}{round(v, 2):g}" for e, v in zip(X.columns, row) if v > 0.1)
                        for row in values]
            result = pd.DataFrame({"formula": formulas, "Tc": labels,
                                   "type": np.where(labels == 0, "synthetic", "original")})
    write_csv(result, args.output)
    print(f"Saved {len(result)} rows; method={args.method}")


if __name__ == "__main__":
    main()

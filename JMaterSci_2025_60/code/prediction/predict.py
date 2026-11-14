"""Batch prediction with explicit formula-column and model selection."""
import argparse
from pathlib import Path

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, MODELS, TABLES, features, load_model, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(DATA / "inputs/alloy_prediction.csv"))
    p.add_argument("--column", default="formula")
    p.add_argument("--model", choices=["ET", "RF"], default="ET")
    p.add_argument("--model-dir", type=Path, default=MODELS)
    p.add_argument("--output", type=Path, default=TABLES / "generated/prediction.csv")
    args = p.parse_args()
    if Path(args.input).resolve() == args.output.resolve():
        p.error("Input must not be overwritten")
    model, columns = load_model(args.model, args.model_dir)
    frame = pd.read_csv(args.input)
    frame["Predicted_Tc"] = model.predict(features(frame[args.column], columns))
    write_csv(frame, args.output)


if __name__ == "__main__":
    main()

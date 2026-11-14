"""Seeded batch screening: all metals, sampling convergence, or Ga-In-Sn."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DATA, MODELS, TABLES, features, load_model, write_csv


def generate_formulas(elements, count, rng):
    if len(elements) == 1:
        return [f"{elements[0]}1.000"]
    formulas = []
    while len(formulas) < count:
        amounts = rng.random((max(32, count - len(formulas)), len(elements)))
        amounts = np.round(amounts / amounts.sum(axis=1, keepdims=True), 3)
        for row in amounts[np.isclose(amounts.sum(axis=1), 1.0)]:
            formulas.append("".join(f"{e}{v:.3f}" for e, v in zip(elements, row) if v != 0))
            if len(formulas) == count:
                break
    return formulas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["alloys", "sampling", "gain-sn"], default="gain-sn")
    p.add_argument("--model", choices=["ET", "RF"], default="ET")
    p.add_argument("--model-dir", type=Path, default=MODELS)
    p.add_argument("--elements", nargs="+")
    p.add_argument("--components", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=TABLES / "generated/screening.csv")
    args = p.parse_args()
    if args.samples < 1 or args.top_k < 1:
        p.error("--samples and --top-k must be positive")
    if args.elements:
        elements = args.elements
    elif args.mode == "gain-sn":
        elements = ["Ga", "In", "Sn"]
    elif args.mode == "sampling":
        elements = ["Ga", "Bi", "In", "Sn", "Zn", "Ag", "Sb", "Cu"]
    else:
        table = pd.read_csv(DATA / "inputs/alloy_element.csv")
        elements = table.loc[table["type"] == "Metal", "Element"].tolist()
    if len(set(elements)) != len(elements):
        p.error("Duplicate elements")
    model, columns = load_model(args.model, args.model_dir)
    if set(elements) - set(columns):
        p.error(f"Unknown model elements: {sorted(set(elements) - set(columns))}")
    rng = np.random.default_rng(args.seed)
    sizes = [len(elements)] if args.mode == "gain-sn" else range(1, 4) if args.mode == "sampling" else [args.components]
    rows = []
    for size in sizes:
        for system in itertools.combinations(elements, size):
            formulas = generate_formulas(system, args.samples, rng)
            predictions = model.predict(features(formulas, columns))
            frame = pd.DataFrame({"elements": "-".join(system), "formula": formulas, "Predicted_Tc": predictions})
            keep = frame if args.mode == "gain-sn" else frame.nlargest(args.top_k if args.mode == "sampling" else 1, "Predicted_Tc")
            rows.extend(keep.to_dict("records"))
            write_csv(pd.DataFrame(rows), args.output)
            print("-".join(system), float(frame.Predicted_Tc.max()), flush=True)


if __name__ == "__main__":
    main()

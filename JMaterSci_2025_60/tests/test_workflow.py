import csv
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/prediction"))
sys.path.insert(0, str(ROOT / "code/modeling"))
sys.path.insert(0, str(ROOT / "code/data"))
import numpy as np
from common import features, parse_formula
from screen import generate_formulas
from train import build_model
from tune import GRID
from sklearn.model_selection import ParameterGrid
from prepare_data import atomic_masses, clean_formula


class WorkflowTests(unittest.TestCase):
    def test_tuning_grid_uses_unique_criteria(self):
        self.assertEqual(GRID["criterion"], ["squared_error"])
        self.assertEqual(len(ParameterGrid(GRID)), 135)

    def test_atomic_mass_conversion(self):
        self.assertEqual(len(atomic_masses()), 118)
        self.assertEqual(atomic_masses()["Ga"], 69.723)
        self.assertEqual(clean_formula("Ga50In50"), ("Ga0.622In0.378", True))
        self.assertEqual(clean_formula("Ga50In50", False), ("Ga50In50", False))
        with self.assertRaises(ValueError):
            clean_formula("Xx50Ga50")

    def test_parser(self):
        self.assertEqual(parse_formula("Nb3Sn"), {"Nb": 3, "Sn": 1})
        self.assertEqual(parse_formula("Fe0Mo1"), {"Mo": 1})
        self.assertEqual(parse_formula("Fe1Fe2"), {"Fe": 3})
        with self.assertRaises(ValueError):
            parse_formula("Fe1-xSe")

    def test_features_keep_raw_ratios(self):
        frame = features(["Nb3Sn"], ["Nb", "Sn"])
        self.assertEqual(frame.iloc[0].tolist(), [3, 1])
        with self.assertRaises(ValueError):
            features(["MgB2"], ["Nb", "Sn"])

    def test_sampling_is_seeded(self):
        a = generate_formulas(["Ga", "In", "Sn"], 50, np.random.default_rng(42))
        b = generate_formulas(["Ga", "In", "Sn"], 50, np.random.default_rng(42))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 50)
        self.assertTrue(all(np.isclose(sum(parse_formula(f).values()), 1) for f in a))

    def test_paper_parameters(self):
        model = build_model("ET")
        self.assertEqual(model.n_estimators, 300)
        self.assertEqual(model.min_samples_split, 5)
        self.assertIn(model.criterion, ("friedman_mse", "squared_error"))

    def test_frozen_artifacts(self):
        with (ROOT / "docs/provenance/artifact_manifest.csv").open() as handle:
            for row in csv.DictReader(handle):
                with self.subTest(path=row["path"]):
                    self.assertEqual(hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest(), row["sha256"])

    def test_data_rows(self):
        with (ROOT / "data/processed/mdr_clean.csv").open() as handle:
            self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 20995)

    def test_corrected_tc(self):
        for name in ("data/source/mdr.csv", "data/processed/mdr_clean.csv"):
            with self.subTest(path=name), (ROOT / name).open() as handle:
                rows = [row for row in csv.DictReader(handle) if row["formula"] == "Ca3Ir3.64Co0.36Sn13"]
                self.assertEqual(len(rows), 1)
                self.assertEqual(float(rows[0]["Tc"]), 7.5)


if __name__ == "__main__":
    unittest.main()

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

# Legacy scripts require the old schema (formula,Tc); never point them at a
# table where Tc is the technetium feature and Tc_K is the target.
DATA_CLEAN = os.path.join(DATA_DIR, "mdr_clean.csv")
DATA_CLEAN_OD2 = os.path.join(DATA_DIR, "mdr_clean_od2.csv")

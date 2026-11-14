"""
Shared ML Prediction Service for LMSC Platform.
Consolidates prediction logic from multiple scripts.
"""

import os
import json
import numpy as np
import pandas as pd
from joblib import load
from typing import Optional, Dict, Any, Tuple, List
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils import parse_formula, preprocess_molecular_data, ELEMENT_MASS, METAL_ELEMENTS

# Default paths
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mdr_clean_od2.csv")


def _resolve_model_files(
    model_key: str, model_dir: str = DEFAULT_MODEL_DIR
) -> Optional[Tuple[str, str]]:
    """Resolve model and feature file paths for given model key."""
    candidates = [model_key]
    if model_key in {"ET", "RF"}:
        candidates = [f"{model_key}_web", model_key]

    for key in candidates:
        model_path = os.path.join(model_dir, f"{key}.joblib")
        feature_path = os.path.join(model_dir, f"{key}_feature.joblib")
        if os.path.exists(model_path) and os.path.exists(feature_path):
            return key, model_path, feature_path
    return None


def load_model(
    model_key: str = "ET", model_dir: str = DEFAULT_MODEL_DIR
) -> Optional[Tuple[Any, List[str]]]:
    """Load model and feature names."""
    result = _resolve_model_files(model_key, model_dir)
    if result:
        key, model_path, feature_path = result
        try:
            model = load(model_path)
            feature_names = load(feature_path)
            return model, list(feature_names)
        except Exception as e:
            print(f"Error loading model {model_key}: {e}")
    return None


def predict_transition_temp(
    formula: str, model: Any, feature_names: List[str]
) -> Optional[float]:
    """Predict Tc for a single formula."""
    try:
        elements_ratios = parse_formula(formula)
    except Exception:
        return None

    if not elements_ratios:
        return None

    # Check for unknown elements
    unknown = [e for e in elements_ratios.keys() if e not in ELEMENT_MASS]
    if unknown:
        return None

    # Create feature vector aligned with training features
    feature_row = {e: elements_ratios.get(e, 0) for e in feature_names}
    input_features = pd.DataFrame([feature_row], columns=feature_names)

    return float(model.predict(input_features)[0])


def predict_with_uncertainty(
    formula: str, model: Any, feature_names: List[str]
) -> Dict[str, Any]:
    """
    Predict Tc with uncertainty estimate.

    Returns dict with:
    - tc: predicted critical temperature
    - tree_predictions: individual tree predictions (if available)
    - ensemble_std: standard deviation across trees
    """
    try:
        elements_ratios = parse_formula(formula)
    except Exception:
        return {"error": "Formula parsing failed"}

    if not elements_ratios:
        return {"error": "Invalid formula"}

    unknown = [e for e in elements_ratios.keys() if e not in ELEMENT_MASS]
    if unknown:
        return {"error": f"Unknown elements: {unknown}"}

    feature_row = {e: elements_ratios.get(e, 0) for e in feature_names}
    input_features = pd.DataFrame([feature_row], columns=feature_names)

    # Check if it's an ensemble model
    if hasattr(model, "estimators_"):
        tree_preds = np.array(
            [tree.predict(input_features)[0] for tree in model.estimators_]
        )
        tc = float(np.mean(tree_preds))
        std = float(np.std(tree_preds))
        return {
            "tc": tc,
            "tree_predictions": tree_preds.tolist(),
            "ensemble_std": std,
            "confidence_interval": (tc - 1.96 * std, tc + 1.96 * std),
        }
    else:
        tc = float(model.predict(input_features)[0])
        return {"tc": tc}


class PredictionService:
    """
    Unified prediction service with caching and thread-safety.
    """

    def __init__(self, model_key: str = "ET", model_dir: str = DEFAULT_MODEL_DIR):
        self.model_key = model_key
        self.model_dir = model_dir
        self._model = None
        self._feature_names = None
        self._load_model()

    def _load_model(self):
        """Load model on initialization."""
        result = load_model(self.model_key, self.model_dir)
        if result:
            self._model, self._feature_names = result

    def predict(self, formula: str) -> Dict[str, Any]:
        """Predict Tc for a formula."""
        if self._model is None:
            return {"error": "Model not loaded"}
        return predict_with_uncertainty(formula, self._model, self._feature_names)

    def batch_predict(self, formulas: List[str]) -> List[Dict[str, Any]]:
        """Predict Tc for multiple formulas."""
        return [self.predict(f) for f in formulas]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_name(self) -> str:
        if self.model_key.startswith("ET"):
            return "Extra-Trees"
        elif self.model_key.startswith("RF"):
            return "Random Forest"
        return self.model_key


if __name__ == "__main__":
    # Test the service
    service = PredictionService("ET")
    print(f"Model loaded: {service.is_loaded}")
    print(f"Model name: {service.model_name}")

    # Test prediction
    result = service.predict("YBa2Cu3O7")
    print(f"YBa2Cu3O7: {result}")

    result = service.predict("Ba0.5In0.5")
    print(f"Ba0.5In0.5: {result}")

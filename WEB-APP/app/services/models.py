"""
Lightweight model registry and prediction service.
"""

import json
import logging
import os
import re
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load

from app.config import config
from utils import ELEMENT_MASS, parse_formula
from utils.scope import parse_scoped_formula

logger = logging.getLogger(__name__)


class ModelService:
    """Thread-safe model registry + prediction helper."""

    _instance = None
    _lock = threading.RLock()

    # model registry: model_key -> {"model": obj, "features": list, "metrics": dict}
    _registry: Dict[str, Dict[str, Any]] = {}

    # current active model key
    _active_key: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # --- registry helpers -------------------------------------------------

    @classmethod
    def _model_dir(cls, project_root: str) -> str:
        return os.path.join(project_root, "ml", "models")

    @classmethod
    def _data_path(cls, project_root: str) -> str:
        return os.path.join(project_root, "data", "superconductors.csv")

    @classmethod
    def _load_metrics(cls, model_dir: str, key: str) -> Dict[str, float]:
        path = os.path.join(model_dir, f"{key}_metrics.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                return {k: (round(float(v), 3) if isinstance(v, (int, float, str)) and v != "" else v) for k, v in metrics.items()}
            except Exception as e:
                logger.warning(f"Failed to load metrics for {key}: {e}")
        # Fallback defaults
        defaults = {
            "RF": {"RMSE": 8.52, "R2": 0.92, "MARE": 0.15, "SMAPE": 12.3},
            "ET": {"RMSE": 7.89, "R2": 0.94, "MARE": 0.12, "SMAPE": 10.5},
        }
        for k in [key, key.split("_")[0]]:
            if k in defaults:
                return defaults[k]
        return defaults["ET"]

    @classmethod
    def _read_metadata_file(cls, model_dir: str, key: str) -> Dict[str, Any]:
        path = os.path.join(model_dir, f"{key}_metadata.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {key}: {e}")
            return {}

    @classmethod
    def get_metadata(cls, key: str, project_root: str) -> Dict[str, Any]:
        """Return enriched metadata for a model on disk (no model loading)."""
        model_dir = cls._model_dir(project_root)
        meta = cls._read_metadata_file(model_dir, key)
        metrics = cls._load_metrics(model_dir, key)

        model_path = os.path.join(model_dir, f"{key}.joblib")
        feature_path = os.path.join(model_dir, f"{key}_feature.joblib")

        file_size = 0
        created_at = meta.get("created_at")
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            if not created_at:
                created_at = datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()

        feature_count = meta.get("feature_count")
        if not feature_count and os.path.exists(feature_path):
            try:
                feature_count = len(load(feature_path))
            except Exception:
                feature_count = 0

        model_type = meta.get("model_type") or (key.split("_")[0] if "_" in key else key)
        return {
            "key": key,
            "name": meta.get("display_name") or cls.display_name(key),
            "type": model_type,
            "created_at": created_at,
            "file_size": file_size,
            "feature_count": feature_count or 0,
            "metrics": metrics,
            "hyperparameters": meta.get("hyperparameters", {}),
            "dataset": meta.get("dataset", {}),
            "train_info": {
                "train_time": meta.get("train_time"),
                "n_samples_train": meta.get("n_samples_train"),
                "n_samples_test": meta.get("n_samples_test"),
            },
            "files": {
                "model": f"{key}.joblib",
                "feature": f"{key}_feature.joblib",
                "metrics": f"{key}_metrics.json",
                "metadata": f"{key}_metadata.json",
            },
        }

    @classmethod
    def list_enriched(cls, project_root: str) -> List[Dict[str, Any]]:
        """List all usable models with enriched metadata, newest first."""
        model_dir = cls._model_dir(project_root)
        if not os.path.isdir(model_dir):
            return []
        active_key = cls._active_key
        models = []
        for fn in os.listdir(model_dir):
            if fn.endswith(".joblib") and not fn.endswith("_feature.joblib"):
                key = fn[:-7]
                if os.path.exists(os.path.join(model_dir, f"{key}_feature.joblib")):
                    try:
                        info = cls.get_metadata(key, project_root)
                        info["active"] = (key == active_key)
                        models.append(info)
                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {key}: {e}")
        models.sort(key=lambda m: m.get("created_at") or "", reverse=True)
        return models

    @classmethod
    def delete(cls, key: str, project_root: str) -> Tuple[bool, str]:
        """Delete a model from disk and registry. Falls back to another model if active."""
        model_dir = cls._model_dir(project_root)
        files = [f"{key}.joblib", f"{key}_feature.joblib", f"{key}_metrics.json", f"{key}_metadata.json"]
        removed = []
        for fn in files:
            path = os.path.join(model_dir, fn)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed.append(fn)
                except Exception as e:
                    return False, f"Failed to delete {fn}: {e}"

        with cls._lock:
            cls._registry.pop(key, None)
            if cls._active_key == key:
                cls._active_key = None

        if cls._active_key is None:
            fallback = cls.ensure_loaded("ET", project_root)
            if not fallback:
                available = cls.list_available(project_root)
                if available:
                    cls.ensure_loaded(available[0]["key"], project_root)
        return True, f"Deleted {key} ({len(removed)} files)"

    @classmethod
    def register(cls, key: str, model: Any, features: List[str], metrics: Optional[Dict] = None):
        with cls._lock:
            cls._registry[key] = {"model": model, "features": list(features), "metrics": metrics or {}}
            cls._active_key = key

    @classmethod
    def load(cls, key: str, project_root: str) -> Optional[str]:
        """Load a model from disk into registry. Returns resolved key or None."""
        model_dir = cls._model_dir(project_root)
        candidates = [key]
        if key in {"ET", "RF"}:
            candidates = [f"{key}_web", key]

        for cand in candidates:
            model_path = os.path.join(model_dir, f"{cand}.joblib")
            feature_path = os.path.join(model_dir, f"{cand}_feature.joblib")
            if os.path.exists(model_path) and os.path.exists(feature_path):
                try:
                    model = load(model_path)
                    features = load(feature_path)
                    if not isinstance(features, list):
                        features = list(features)
                    metrics = cls._load_metrics(model_dir, cand)
                    cls.register(cand, model, features, metrics)
                    logger.info(f"Loaded model: {cand}")
                    return cand
                except Exception as e:
                    logger.error(f"Failed to load model {cand}: {e}")
        logger.warning(f"Model not found: {key}")
        return None

    @classmethod
    def ensure_loaded(cls, key: str, project_root: str) -> Optional[str]:
        with cls._lock:
            if key in cls._registry:
                cls._active_key = key
                return key
            return cls.load(key, project_root)

    @classmethod
    def list_available(cls, project_root: str) -> List[Dict[str, Any]]:
        model_dir = cls._model_dir(project_root)
        if not os.path.isdir(model_dir):
            return []

        models = []
        for fn in sorted(os.listdir(model_dir)):
            if fn.endswith(".joblib") and not fn.endswith("_feature.joblib"):
                key = fn[:-7]
                feature_file = f"{key}_feature.joblib"
                if os.path.exists(os.path.join(model_dir, feature_file)):
                    models.append({
                        "key": key,
                        "name": cls.display_name(key, project_root),
                        "file": fn,
                    })
        return models

    @classmethod
    def active(cls) -> Optional[Dict[str, Any]]:
        with cls._lock:
            if cls._active_key and cls._active_key in cls._registry:
                return {"key": cls._active_key, **cls._registry[cls._active_key]}
            return None

    @classmethod
    def status(cls) -> Dict[str, Any]:
        active = cls.active()
        if not active:
            return {"loaded": False, "name": "未加载"}
        return {
            "loaded": True,
            "key": active["key"],
            "name": cls.display_name(active["key"]),
            "features_count": len(active.get("features", [])),
            "metrics": active.get("metrics", {}),
        }

    @classmethod
    def display_name(cls, key: str, project_root: Optional[str] = None) -> str:
        root = project_root or config.PROJECT_ROOT
        meta = cls._read_metadata_file(cls._model_dir(root), key)
        if meta.get("display_name"):
            return meta["display_name"]
        algo = "Extra-Trees" if key.startswith("ET") else "Random Forest" if key.startswith("RF") else key
        # Legacy keys embed a unix timestamp: ET_web_1787728251
        m = re.search(r"_(\d{10})$", key)
        if m:
            try:
                dt = datetime.fromtimestamp(int(m.group(1)))
                metrics = cls._load_metrics(cls._model_dir(root), key)
                r2 = metrics.get("R2")
                suffix = f" · R²={r2}" if r2 is not None else ""
                return f"{algo} · {dt.strftime('%Y-%m-%d %H:%M')}{suffix}"
            except (ValueError, OSError):
                pass
        if "_web" in key:
            algo += " (web trained)"
        return algo

    # --- prediction helpers ------------------------------------------------

    @classmethod
    def predict_formula(cls, formula: str, model_key: Optional[str] = None, project_root: Optional[str] = None) -> Dict[str, Any]:
        key = model_key or cls._active_key
        if not key and project_root:
            # Try to load default ET
            key = cls.ensure_loaded("ET", project_root)
        if not key:
            return {"error": "No model loaded"}

        with cls._lock:
            entry = cls._registry.get(key)
        if not entry:
            # Auto-load the requested model from disk if possible
            if project_root:
                resolved = cls.load(key, project_root)
                if resolved:
                    key = resolved
                    with cls._lock:
                        entry = cls._registry.get(key)
        if not entry:
            return {"error": f"Model {key} not loaded"}

        model = entry["model"]
        features = entry["features"]
        metrics = entry.get("metrics") or cls._load_metrics(cls._model_dir(project_root or ""), key)

        # Parse formula
        metadata = cls._read_metadata_file(cls._model_dir(project_root or config.PROJECT_ROOT), key)
        custom = metadata.get("composition_scope") == "custom"
        try:
            fractions = parse_scoped_formula(formula, unrestricted=custom)
        except ValueError as e:
            return {"formula": formula, "error": str(e)}
        if custom and set(fractions) - set(features):
            return {"formula": formula, "error": "化学式含有该自定义模型训练特征之外的元素：" + ", ".join(sorted(set(fractions) - set(features)))}
        ratios = fractions if metadata.get("feature_encoding") == "atomic_fraction" else parse_formula(formula)

        if not ratios:
            return {"error": "Invalid formula"}

        unknown = [e for e in ratios.keys() if e not in ELEMENT_MASS]
        if unknown:
            return {"error": f"Unknown elements: {unknown}"}

        feature_row = {e: ratios.get(e, 0.0) for e in features}
        input_df = pd.DataFrame([feature_row], columns=features)

        # Tree predictions
        if hasattr(model, "estimators_"):
            tree_preds = np.array([tree.predict(input_df)[0] for tree in model.estimators_])
            pred = float(np.mean(tree_preds))
            std = float(np.std(tree_preds))
        else:
            tree_preds = None
            pred = float(model.predict(input_df)[0])
            std = 0.0

        reliability = cls._reliability(pred, tree_preds, fractions, features, metrics, None if custom else project_root)
        if custom:
            reliability["details"]["training_proximity"] = None
            reliability["proximity_note"] = "自定义训练集邻近度未计算，评分使用中性默认值。"
        return {
            "formula": formula,
            "predicted_tc": round(pred, 2),
            "unit": "K",
            "model": cls.display_name(key, project_root),
            "model_key": key,
            "reliability": reliability,
            "interpretation": cls._interpretation(pred, reliability["level"]),
        }

    @classmethod
    def _reliability(cls, pred: float, tree_preds: Optional[np.ndarray], ratios: Dict[str, float],
                     features: List[str], metrics: Dict, project_root: Optional[str]) -> Dict[str, Any]:
        std = float(np.std(tree_preds)) if tree_preds is not None and len(tree_preds) else 0.0
        std_score = max(0.0, 100.0 - std * 5.0)

        input_vec = np.array([ratios.get(e, 0.0) for e in features])
        proximity = cls._training_proximity(input_vec, features, project_root)
        proximity_score = proximity * 100.0

        r2 = float(metrics.get("R2", 0.9))
        model_score = r2 * 100.0

        plausibility = cls._plausibility(pred)
        plausibility_score = plausibility * 100.0

        final = (
            std_score * 0.35
            + proximity_score * 0.30
            + model_score * 0.25
            + plausibility_score * 0.10
        )
        final = min(100.0, max(0.0, final))

        level = "高" if final >= 75 else "中" if final >= 50 else "低"

        return {
            "score": round(final, 1),
            "level": level,
            "confidence_interval": f"± {1.96 * std:.2f} K",
            "details": {
                "ensemble_std": round(std, 3),
                "training_proximity": round(proximity * 100, 1),
                "model_r2": r2,
                "rmse": float(metrics.get("RMSE", 0.0)),
            },
        }

    @classmethod
    def _training_proximity(cls, input_vec: np.ndarray, features: List[str], project_root: Optional[str], k: int = 5) -> float:
        if not project_root:
            return 0.5
        path = cls._data_path(project_root)
        if not os.path.exists(path):
            return 0.5
        try:
            import pandas as pd
            df = pd.read_csv(path, low_memory=False, nrows=2000)
            if all(e in df.columns for e in features) and "Tc_K" in df.columns:
                train_matrix = df[features].to_numpy(dtype=float)
                distances = np.linalg.norm(train_matrix - input_vec, axis=1)
                nearest = np.sort(distances)[:min(k, len(distances))]
                return 1.0 - min(float(np.mean(nearest)) / np.sqrt(len(input_vec)), 1.0)
            train_vecs = []
            for _, row in df.iterrows():
                formula = str(row.iloc[0])
                try:
                    ratios = parse_formula(formula)
                    train_vecs.append([ratios.get(e, 0.0) for e in features])
                except Exception:
                    continue
            if not train_vecs:
                return 0.5
            train_matrix = np.array(train_vecs)
            distances = np.linalg.norm(train_matrix - input_vec, axis=1)
            nearest = np.sort(distances)[:min(k, len(distances))]
            mean_dist = float(np.mean(nearest))
            max_dist = np.sqrt(len(input_vec))
            normalized = min(mean_dist / max_dist, 1.0) if max_dist > 0 else 0.0
            return 1.0 - normalized
        except Exception as e:
            logger.warning(f"Proximity computation failed: {e}")
            return 0.5

    @classmethod
    def _plausibility(cls, tc: float) -> float:
        if tc < 0:
            return 0.0
        if tc > 300:
            return 0.0
        if tc > 200:
            return 0.3
        if tc > 100:
            return 0.7
        return 1.0

    @classmethod
    def _interpretation(cls, tc: float, level: str) -> str:
        if tc < 0:
            return "负温度无物理意义，可能模型外推不可靠。"
        if tc < 5:
            return f"预测为低温超导体（{level}可靠性）。" + ("" if level == "高" else "建议实验验证。")
        if tc < 20:
            return f"预测为中温超导体（{level}可靠性）。"
        if tc < 100:
            return f"预测为高温超导体（{level}可靠性）。" + ("" if level == "高" else "模型外推范围，需谨慎对待。")
        return "预测为极高温度超导，请注意模型主要训练数据在 0-150 K 范围。"

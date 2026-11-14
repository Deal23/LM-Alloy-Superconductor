"""
Lightweight training service.
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
from joblib import dump

from app.services.models import ModelService
from utils import preprocess_molecular_data
from utils.scope import ELEMENTS, parse_scoped_formula

logger = logging.getLogger(__name__)

DEFAULT_DATASET_MARKER = "superconductors"


def _dataset_slug(data_path: str) -> str:
    """Return a short safe slug for the training dataset, or '' for the default dataset."""
    if not data_path or DEFAULT_DATASET_MARKER in os.path.basename(data_path):
        return ""
    base = os.path.splitext(os.path.basename(data_path))[0]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_")
    return slug[:20]


def build_model_key(
    model_type: str,
    n_estimators: int,
    max_depth,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features,
    data_path: str,
) -> str:
    """Build a descriptive, distinguishable model key.

    Example: ET_web_20260826_203015_n300_mdAuto_ms2_ml1_mfsqrt_dsMyAlloy
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md = max_depth if max_depth not in (None, "", "None", 0) else "Auto"
    mf = max_features if max_features not in (None, "", "None") else "Auto"
    params = f"n{n_estimators}_md{md}_ms{min_samples_split}_ml{min_samples_leaf}_mf{mf}"
    slug = _dataset_slug(data_path)
    suffix = f"_ds{slug}" if slug else ""
    return f"{model_type}_web_{ts}_{params}{suffix}"


def build_display_name(
    model_type: str,
    n_estimators: int,
    data_path: str,
    created_at: datetime,
) -> str:
    algo = "Extra-Trees" if model_type == "ET" else "Random Forest"
    dt = created_at.strftime("%Y-%m-%d %H:%M")
    parts = [f"{algo} · {dt} · n={n_estimators}"]
    slug = _dataset_slug(data_path)
    if slug:
        parts.append(f"数据集 {slug}")
    else:
        parts.append("默认数据集")
    return " · ".join(parts)


class TrainingService:
    """Train tree-based models and register them on the fly."""

    @classmethod
    def train(
        cls,
        project_root: str,
        data_path: str,
        model_type: str = "ET",
        test_size: float = 0.2,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[str] = "sqrt",
        display_name: Optional[str] = None,
        progress=None,
    ) -> Tuple[bool, Dict[str, Any]]:
        try:
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
        except ImportError as e:
            logger.error(f"scikit-learn not installed: {e}")
            return False, {"error": "scikit-learn is required for training"}

        import pandas as pd

        try:
            if not os.path.exists(data_path):
                return False, {"error": f"Data file not found: {data_path}"}

            if data_path.endswith(".xlsx"):
                df = pd.read_excel(data_path)
            else:
                df = pd.read_csv(data_path, low_memory=False)

            if progress:
                progress("准备数据", 5)
            builtin = os.path.basename(data_path) == "superconductors.csv" and "Tc_K" in df.columns and ELEMENTS.issubset(df.columns)
            if builtin:
                for formula in df["formula"]:
                    parse_scoped_formula(formula, allow_zero=True)
                X = df[sorted(ELEMENTS)].apply(pd.to_numeric, errors="raise")
                y = pd.to_numeric(df["Tc_K"], errors="raise")
                if (X < 0).any().any() or not np.allclose(X.sum(axis=1), 1.0, atol=1e-6):
                    return False, {"error": "原子分数必须非负且每行之和为 1"}
            else:
                if "formula" not in df or "Tc" not in df:
                    return False, {"error": "数据集需要 formula 和 Tc 列（或 LM-based 建模表）"}
                rows = [parse_scoped_formula(formula, unrestricted=True) for formula in df["formula"]]
                X = pd.DataFrame(rows).fillna(0).sort_index(axis=1)
                y = pd.to_numeric(df["Tc"], errors="raise")
            if X.isna().any().any() or y.isna().any():
                return False, {"error": "数据包含缺失或无效数值"}
            if X.empty:
                return False, {"error": "No valid samples after preprocessing"}

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            max_depth_val = None if max_depth in (None, "None", "", 0) else int(max_depth)
            max_features_val = None if max_features in (None, "None", "") else max_features

            ModelClass = ExtraTreesRegressor if model_type == "ET" else RandomForestRegressor
            model = ModelClass(
                n_estimators=int(n_estimators),
                max_depth=max_depth_val,
                min_samples_split=int(min_samples_split),
                min_samples_leaf=int(min_samples_leaf),
                max_features=max_features_val,
                random_state=42,
                n_jobs=-1,
                warm_start=True,
            )

            start = time.time()
            total = int(n_estimators)
            step = max(1, min(20, total // 10 or 1))
            for count in range(step, total + step, step):
                model.set_params(n_estimators=min(count, total))
                model.fit(X_train, y_train)
                if progress:
                    progress(f"训练决策树 {min(count, total)}/{total}", 10 + int(75 * min(count, total) / total))
            train_time = time.time() - start

            y_pred = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))

            features = list(X.columns)
            created_at = datetime.now()
            model_key = build_model_key(
                model_type=model_type,
                n_estimators=int(n_estimators),
                max_depth=max_depth,
                min_samples_split=int(min_samples_split),
                min_samples_leaf=int(min_samples_leaf),
                max_features=max_features,
                data_path=data_path,
            ) + "_" + uuid.uuid4().hex[:8]
            display_name = (display_name or "").strip() or build_display_name(
                model_type=model_type,
                n_estimators=int(n_estimators),
                data_path=data_path,
                created_at=created_at,
            )
            if progress:
                progress("评估并保存模型", 90)

            model_dir = os.path.join(project_root, "ml", "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model_key}.joblib")
            feature_path = os.path.join(model_dir, f"{model_key}_feature.joblib")
            dump(model, model_path)
            dump(features, feature_path)

            metrics = {"RMSE": round(rmse, 3), "R2": round(r2, 3)}
            metrics_path = os.path.join(model_dir, f"{model_key}_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            rel_data_path = (
                os.path.relpath(data_path, project_root)
                if os.path.abspath(data_path).startswith(os.path.abspath(project_root) + os.sep)
                else os.path.basename(data_path)
            )
            metadata = {
                "model_key": model_key,
                "model_type": model_type,
                "display_name": display_name,
                "created_at": created_at.isoformat(),
                "data_path": rel_data_path,
                "dataset": {
                    "path": rel_data_path,
                    "name": _dataset_slug(data_path) or "默认数据集",
                },
                "hyperparameters": {
                    "n_estimators": int(n_estimators),
                    "max_depth": max_depth_val,
                    "min_samples_split": int(min_samples_split),
                    "min_samples_leaf": int(min_samples_leaf),
                    "max_features": max_features_val,
                    "test_size": float(test_size),
                    "random_state": 42,
                },
                "metrics": metrics,
                "feature_count": len(features),
                "feature_encoding": "atomic_fraction",
                "composition_scope": "builtin" if builtin else "custom",
                "train_time": round(train_time, 2),
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
            }
            metadata_path = os.path.join(model_dir, f"{model_key}_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            ModelService.register(model_key, model, features, metrics)

            return True, {
                "model_key": model_key,
                "name": display_name,
                "metrics": metrics,
                "train_time": round(train_time, 2),
                "n_estimators": n_estimators,
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
                "hyperparameters": metadata["hyperparameters"],
                "dataset": metadata["dataset"],
                "created_at": metadata["created_at"],
            }
        except Exception as e:
            logger.exception("Training failed")
            return False, {"error": str(e)}

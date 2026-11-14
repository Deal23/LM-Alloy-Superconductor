"""Tree-model registry and default/search configurations for the Tc study."""
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

SEED = 42

DEFAULT_PARAMS = {
    "ExtraTrees": dict(
        n_estimators=300,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=1.0,
        n_jobs=-1,
    ),
    "RandomForest": dict(
        n_estimators=300,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features=1.0,
        n_jobs=-1,
    ),
    "HistGradientBoosting": dict(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=False,
    ),
    "XGBoost": dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
    ),
    "LightGBM": dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        verbosity=-1,
    ),
    "CatBoost": dict(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    ),
}

SEARCH_SPACES = {
    "ExtraTrees": {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", 0.5],
    },
    "RandomForest": {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt", 0.5],
    },
    "HistGradientBoosting": {
        "max_iter": [200, 400, 800],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [5, 10, 20],
        "l2_regularization": [0.0, 1.0, 10.0],
    },
    "XGBoost": {
        "n_estimators": [300, 500, 800, 1200],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [3, 4, 6, 8],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0],
    },
    "LightGBM": {
        "n_estimators": [300, 500, 800, 1200],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [5, 10, 20],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0],
    },
    "CatBoost": {
        "iterations": [400, 700, 1000],
        "learning_rate": [0.03, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1.0, 3.0, 10.0],
    },
}

AVAILABLE_MODELS = ["ExtraTrees", "RandomForest", "HistGradientBoosting"]
if HAS_XGBOOST:
    AVAILABLE_MODELS.append("XGBoost")
if HAS_LIGHTGBM:
    AVAILABLE_MODELS.append("LightGBM")
if HAS_CATBOOST:
    AVAILABLE_MODELS.append("CatBoost")


def build_model(name, params=None, seed=SEED):
    if name not in DEFAULT_PARAMS:
        raise ValueError(f"unknown model: {name}")
    kwargs = dict(DEFAULT_PARAMS[name])
    if params:
        kwargs.update(params)

    if name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=seed, **kwargs)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=seed, **kwargs)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=seed, **kwargs)
    if name == "XGBoost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed")
        return XGBRegressor(random_state=seed, **kwargs)
    if name == "LightGBM":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is not installed")
        return LGBMRegressor(random_state=seed, **kwargs)
    if name == "CatBoost":
        if not HAS_CATBOOST:
            raise ImportError("catboost is not installed")
        kwargs.setdefault("random_seed", seed)
        return CatBoostRegressor(**kwargs)
    raise ValueError(f"unknown model: {name}")

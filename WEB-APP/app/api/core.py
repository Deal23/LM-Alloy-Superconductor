"""
Core API blueprint for the lightweight LMSC web app.
"""

import glob
import json
import logging
import os
import threading
import uuid
import time
from typing import Any, Dict

import pandas as pd
from flask import Blueprint, jsonify, request, send_file

from app.config import config
from app.services.models import ModelService
from app.services.tasks import TaskService
from app.services.training import TrainingService

logger = logging.getLogger(__name__)

core_bp = Blueprint("core", __name__, url_prefix="/api")
_training_jobs = {}
_training_lock = threading.Lock()


@core_bp.route("/resources", methods=["GET"])
def resources():
    from app.services.resources import system_resources
    return jsonify(system_resources())


def _local_update_request():
    from urllib.parse import urlsplit
    from app.services.updates import is_local
    if not is_local(request.remote_addr) or request.host.split(":")[0] not in {"localhost", "127.0.0.1"}:
        return False
    origin = request.headers.get("Origin")
    return not origin or (urlsplit(origin).netloc == request.host and urlsplit(origin).scheme in {"http", "https"})


@core_bp.route("/updates", methods=["GET"])
def check_updates():
    if not _local_update_request():
        return jsonify({"error": "更新功能仅允许本机访问"}), 403
    try:
        from app.services.updates import latest_release
        info = latest_release()
        return jsonify(info)
    except Exception as exc:
        logger.warning("Update check failed: %s", exc)
        return jsonify({"error": "无法检查更新，请检查网络或稍后重试"}), 502


@core_bp.route("/updates/apply", methods=["POST"])
def update_app():
    if not _local_update_request() or not request.is_json:
        return jsonify({"error": "更新功能仅允许本机页面操作"}), 403
    with _training_lock:
        if any(job.get("status") in {"queued", "running"} for job in _training_jobs.values()):
            return jsonify({"error": "请等待训练任务结束后再更新"}), 409
    with TaskService._lock:
        if any(status in {"queued", "running"} for status in TaskService._status.values()):
            return jsonify({"error": "请等待筛选任务结束后再更新"}), 409
    from ml.visualization.phase_jobs import JOBS, LOCK
    with LOCK:
        if any(job.get("status") in {"queued", "running"} for job in JOBS.values()):
            return jsonify({"error": "请等待相图任务结束后再更新"}), 409
    try:
        from app.services.updates import apply_update
        result = apply_update(config.PROJECT_ROOT, (request.get_json(silent=True) or {}).get("version"))
        return jsonify({"success": True, **result})
    except Exception as exc:
        logger.warning("Update failed: %s", exc)
        return jsonify({"error": "更新失败，已尝试恢复原文件；请检查网络和远端代码"}), 400


def _default_data_path() -> str:
    return os.path.join(config.PROJECT_ROOT, "data", "superconductors.csv")


def _resolve_dataset(dataset_id: str) -> str:
    if not dataset_id:
        return _default_data_path()
    from app.api.data_center import _load_metadata

    meta = _load_metadata()
    if dataset_id in meta:
        return os.path.join(config.PROJECT_ROOT, meta[dataset_id]["file_path"])
    return _default_data_path()


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------


@core_bp.route("/health", methods=["GET"])
def health():
    active = ModelService.active()
    model_dir = os.path.join(config.PROJECT_ROOT, "ml", "models")
    model_files = any(f.endswith(".joblib") and not f.endswith("_feature.joblib") for f in os.listdir(model_dir)) if os.path.isdir(model_dir) else False
    data_files = os.path.exists(_default_data_path())
    return jsonify({
        "status": "healthy" if active and data_files else "degraded",
        "model_loaded": active is not None,
        "model_files": model_files,
        "data_files": data_files,
    })


@core_bp.route("/model_status", methods=["GET"])
def model_status():
    return jsonify(ModelService.status())


@core_bp.route("/model_options", methods=["GET"])
def model_options():
    return jsonify({"options": ModelService.list_available(config.PROJECT_ROOT)})


@core_bp.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": ModelService.list_enriched(config.PROJECT_ROOT)})


@core_bp.route("/models/<path:key>/load", methods=["POST"])
def load_model_by_key(key: str):
    resolved = ModelService.ensure_loaded(key, config.PROJECT_ROOT)
    if resolved:
        return jsonify({"success": True, "model_info": ModelService.status()})
    return jsonify({
        "success": False,
        "error": f"Model {key} not found",
        "available": ModelService.list_available(config.PROJECT_ROOT),
    }), 404


@core_bp.route("/models/<path:key>", methods=["DELETE"])
def delete_model(key: str):
    if not os.path.exists(os.path.join(config.PROJECT_ROOT, "ml", "models", f"{key}.joblib")):
        return jsonify({"success": False, "error": f"Model {key} not found"}), 404
    success, message = ModelService.delete(key, config.PROJECT_ROOT)
    if success:
        return jsonify({"success": True, "message": message, "model_info": ModelService.status()})
    return jsonify({"success": False, "error": message}), 500


@core_bp.route("/models/<path:key>/rename", methods=["POST"])
def rename_model(key: str):
    available = {item["key"] for item in ModelService.list_available(config.PROJECT_ROOT)}
    if key not in available:
        return jsonify({"error": "模型不存在"}), 404
    name = (request.get_json(silent=True) or {}).get("name", "")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        return jsonify({"error": "名称不能为空且最多 80 个字符"}), 400
    model_dir = ModelService._model_dir(config.PROJECT_ROOT)
    meta = ModelService._read_metadata_file(model_dir, key)
    meta["display_name"] = name.strip()
    with open(os.path.join(model_dir, f"{key}_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    return jsonify({"success": True, "model_info": ModelService.status()})


@core_bp.route("/models/batch_delete", methods=["POST"])
def batch_delete_models():
    payload = request.get_json(silent=True) or {}
    keys = payload.get("keys", [])
    if not keys or not isinstance(keys, list):
        return jsonify({"success": False, "error": "No keys provided"}), 400

    deleted, failed = [], []
    for key in keys:
        if not os.path.exists(os.path.join(config.PROJECT_ROOT, "ml", "models", f"{key}.joblib")):
            failed.append({"key": key, "error": "not found"})
            continue
        success, message = ModelService.delete(key, config.PROJECT_ROOT)
        if success:
            deleted.append(key)
        else:
            failed.append({"key": key, "error": message})

    return jsonify({
        "success": len(failed) == 0,
        "deleted": deleted,
        "failed": failed,
        "model_info": ModelService.status(),
    })


@core_bp.route("/load_model", methods=["POST"])
def load_model():
    payload = request.get_json(silent=True) or {}
    model_type = payload.get("model_type", "ET")
    resolved = ModelService.ensure_loaded(model_type, config.PROJECT_ROOT)
    if resolved:
        return jsonify({"success": True, "model_info": ModelService.status()})
    return jsonify({
        "success": False,
        "error": f"Model {model_type} not found",
        "available": ModelService.list_available(config.PROJECT_ROOT),
    }), 404


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@core_bp.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    formula = payload.get("formula", "").strip()
    model_key = payload.get("model_key")
    if not formula:
        return jsonify({"error": "No formula provided"}), 400
    result = ModelService.predict_formula(formula, model_key, config.PROJECT_ROOT)
    return jsonify(result)


@core_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    payload = request.get_json(silent=True) or {}
    formulas = payload.get("formulas", [])
    model_key = payload.get("model_key")
    if not formulas or not isinstance(formulas, list):
        return jsonify({"error": "No formulas provided"}), 400
    results = [ModelService.predict_formula(f.strip(), model_key, config.PROJECT_ROOT) for f in formulas]
    return jsonify({"results": results})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@core_bp.route("/train", methods=["POST"])
def train():
    payload = request.get_json(silent=True) or {}
    dataset_id = payload.get("dataset_id")
    data_path = _resolve_dataset(dataset_id)

    success, result = TrainingService.train(
        project_root=config.PROJECT_ROOT,
        data_path=data_path,
        model_type=payload.get("model_type", "ET"),
        test_size=float(payload.get("test_size", 0.2)),
        n_estimators=int(payload.get("n_estimators", 300)),
        max_depth=payload.get("max_depth"),
        min_samples_split=int(payload.get("min_samples_split", 2)),
        min_samples_leaf=int(payload.get("min_samples_leaf", 1)),
        max_features=payload.get("max_features", "sqrt"),
        display_name=payload.get("display_name"),
    )
    if success:
        return jsonify({"success": True, "model_info": result})
    return jsonify({"success": False, "error": result.get("error", "Training failed")}), 500


@core_bp.route("/training_jobs", methods=["POST"])
def start_training_job():
    payload = request.get_json(silent=True) or {}
    try:
        estimators = int(payload.get("n_estimators", 300))
        if not 10 <= estimators <= 2000:
            raise ValueError("树数量必须在 10–2000 之间")
        if payload.get("model_type", "ET") not in {"ET", "RF"}:
            raise ValueError("不支持的算法")
        name = payload.get("display_name", "")
        if not isinstance(name, str) or len(name.strip()) > 80:
            raise ValueError("模型名称不能超过 80 个字符")
        split = int(payload.get("min_samples_split", 2))
        leaf = int(payload.get("min_samples_leaf", 1))
        if split < 2 or leaf < 1:
            raise ValueError("样本数参数无效")
        dataset_id = payload.get("dataset_id")
        if dataset_id:
            from app.api.data_center import _load_metadata
            if dataset_id not in _load_metadata():
                raise ValueError("数据集不存在，请重新选择")
        data_path = _resolve_dataset(dataset_id)
        if not os.path.isfile(data_path):
            raise ValueError("找不到数据文件；请确认 data/superconductors.csv 存在")
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    with _training_lock:
        if any(job["status"] == "running" for job in _training_jobs.values()):
            return jsonify({"error": "已有训练任务正在进行，请等待完成"}), 409
        job_id = uuid.uuid4().hex
        _training_jobs[job_id] = {"status": "running", "stage": "等待训练", "progress": 0, "started_at": time.time(), "eta_seconds": None}

    def update(stage, progress):
        with _training_lock:
            job = _training_jobs[job_id]
            elapsed = time.time() - job["started_at"]
            # Only tree-fitting progress is proportional to work; preprocessing
            # and final model serialization have no reliable duration estimate.
            fraction = (progress - 10) / 75
            eta = elapsed * (1 - fraction) / fraction if 0 < fraction < 1 and stage.startswith("训练决策树") else None
            job.update(stage=stage, progress=progress, elapsed_seconds=round(elapsed, 1), eta_seconds=round(eta) if eta is not None else None)

    def run():
        try:
            success, result = TrainingService.train(
                project_root=config.PROJECT_ROOT, data_path=data_path,
                model_type=payload.get("model_type", "ET"), n_estimators=estimators,
                max_depth=payload.get("max_depth"), min_samples_split=split,
                min_samples_leaf=leaf, display_name=name, progress=update,
            )
            with _training_lock:
                _training_jobs[job_id].update(status="completed" if success else "failed",
                    stage="训练完成" if success else "训练失败", progress=100 if success else _training_jobs[job_id]["progress"],
                    result=result if success else None, error=None if success else result.get("error", "训练失败"))
        except Exception as exc:
            logger.exception("Training job failed")
            with _training_lock:
                _training_jobs[job_id].update(status="failed", stage="训练失败", error=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@core_bp.route("/training_jobs/<job_id>", methods=["GET"])
def training_job_status(job_id):
    with _training_lock:
        job = _training_jobs.get(job_id)
        if not job:
            return jsonify({"error": "训练任务不存在，服务可能已重启"}), 404
        return jsonify(dict(job))


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@core_bp.route("/tasks", methods=["GET"])
def list_tasks():
    return jsonify({"tasks": TaskService.list_scripts()})


@core_bp.route("/run_task", methods=["POST"])
def run_task():
    payload = request.get_json(silent=True) or {}
    task_name = payload.get("task_name")
    params = payload.get("params", {})
    if task_name == "alloy_predict":
        from utils.scope import METALS
        elements = params.get("elements", [])
        if not isinstance(elements, list) or len(set(elements)) != len(elements) or any(e not in METALS for e in elements):
            return jsonify({"error": "高通量筛选仅支持当前范围内的金属元素，且不能重复或包含 O/H"}), 400
        if len(elements) < int(params.get("num_elements", 3)):
            return jsonify({"error": "目标元素数量不足，请调整最大元素数"}), 400
    task_id = TaskService.submit(config.PROJECT_ROOT, task_name, params)
    if task_id is None:
        return jsonify({"error": "Unknown task or script missing"}), 400
    return jsonify({"task_id": task_id, "status": "started"})


@core_bp.route("/task_status/<task_id>", methods=["GET"])
def task_status(task_id: str):
    return jsonify(TaskService.status(task_id))


@core_bp.route("/stop_task/<task_id>", methods=["POST"])
def stop_task(task_id: str):
    return jsonify(TaskService.stop(task_id))


@core_bp.route("/task_result/<task_name>", methods=["GET"])
def task_result(task_name: str):
    """Return the latest CSV result for a known task."""
    if task_name == "alloy_predict":
        pattern = os.path.join(config.PROJECT_ROOT, "results", "predictions", "*Alloy_*.csv")
        files = glob.glob(pattern)
        if not files:
            return jsonify({"ready": False, "error": "No result files found"})
        latest = max(files, key=os.path.getmtime)
        df = pd.read_csv(latest)
        if "Predicted_Tc" in df.columns:
            df = df.sort_values("Predicted_Tc", ascending=False)
        return jsonify({
            "ready": True,
            "file": os.path.basename(latest),
            "total": len(df),
            "columns": list(df.columns),
            "top10": df.head(10).to_dict("records"),
        })

    return jsonify({"ready": False, "error": f"Unknown task: {task_name}"}), 400


@core_bp.route("/task_download/<task_name>", methods=["GET"])
def task_download(task_name: str):
    if task_name == "alloy_predict":
        pattern = os.path.join(config.PROJECT_ROOT, "results", "predictions", "*Alloy_*.csv")
        files = glob.glob(pattern)
        if not files:
            return "No result file found", 404
        latest = max(files, key=os.path.getmtime)
        return send_file(latest, mimetype="text/csv", as_attachment=True, download_name=os.path.basename(latest))
    return "Unknown task", 400


# ---------------------------------------------------------------------------
# Reliability explanation
# ---------------------------------------------------------------------------


@core_bp.route("/reliability_explanation", methods=["GET"])
def reliability_explanation():
    return jsonify({
        "description": "预测可靠度评分基于以下因素综合评估：",
        "factors": [
            {"name": "ensemble_std", "weight": "35%", "description": "树模型预测方差，越小越确定"},
            {"name": "training_proximity", "weight": "30%", "description": "输入配方与训练集的邻近程度"},
            {"name": "model_r2", "weight": "25%", "description": "模型整体拟合质量"},
            {"name": "tc_plausibility", "weight": "10%", "description": "预测温度是否在物理合理范围"},
        ],
        "levels": {
            "高": "分数 ≥ 75，建议相信预测结果",
            "中": "分数 50-75，供参考，建议实验验证",
            "低": "分数 < 50，模型外推，需谨慎",
        },
    })

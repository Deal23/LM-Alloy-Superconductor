import os
import json
import time
import uuid
import math
from collections import Counter
from typing import Any, Dict

import pandas as pd
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from app.config import config
from utils.scope import parse_scoped_formula


data_center_bp = Blueprint("data_center", __name__, url_prefix="/api/dataset")

UPLOAD_DIR = config.UPLOADS_DIR
METADATA_FILE = os.path.join(UPLOAD_DIR, "datasets.json")
BUILTIN_ID = "builtin"


def _builtin_dataset():
    path = os.path.join(config.PROJECT_ROOT, "data", "superconductors.csv")
    available = os.path.isfile(path)
    rows = valid_rows = None
    if available:
        try:
            data = pd.read_csv(path, usecols=["formula", "Tc_K"])
            rows = len(data)
            valid_rows = int(data["Tc_K"].count())
        except (OSError, ValueError, pd.errors.ParserError):
            available = False
    return {
        "id": BUILTIN_ID, "name": "LM 基础数据",
        "file_path": "data/superconductors.csv", "builtin": True,
        "rows": rows, "valid_rows": valid_rows, "available": available,
        "description": "不含 O/H、至少含一种金属的超导材料组成与临界温度数据。",
    }


def _load_metadata() -> Dict[str, Any]:
    path = METADATA_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _save_metadata(data: Dict[str, Any]):
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@data_center_bp.route("/list", methods=["GET"])
def list_datasets():
    metadata = _load_metadata()
    datasets = sorted(metadata.values(), key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"success": True, "datasets": [_builtin_dataset(), *datasets]})


@data_center_bp.route("/<dataset_id>/analysis", methods=["GET"])
def analyze_dataset(dataset_id):
    builtin = dataset_id == BUILTIN_ID
    metadata = _load_metadata()
    if not builtin and dataset_id not in metadata:
        return jsonify({"error": "数据集不存在"}), 404
    info = _builtin_dataset() if builtin else metadata[dataset_id]
    path = os.path.join(config.PROJECT_ROOT, info["file_path"])
    if not os.path.isfile(path):
        return jsonify({"error": "数据文件不存在"}), 404
    try:
        threshold = request.args.get("threshold", default=0, type=float)
        if threshold is None or not math.isfinite(threshold) or threshold < 0:
            return jsonify({"error": "Tc 阈值必须为非负数"}), 400
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        target = "Tc_K" if builtin else "Tc"
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df[df[target].notna() & (df[target] >= threshold)]
        if df.empty:
            return jsonify({"name": info["name"], "count": 0, "histogram": [], "elements": [], "components": []})
        counts, components = Counter(), Counter()
        for formula in df["formula"]:
            fractions = parse_scoped_formula(formula, allow_zero=builtin, unrestricted=not builtin)
            counts.update(fractions.keys())
            components[len(fractions)] += 1
        values = df[target].to_numpy(dtype=float)
        low, high = float(values.min()), float(values.max())
        bins = min(12, max(1, len(values))) if high > low else 1
        import numpy as np
        frequencies, edges = np.histogram(values, bins=bins)
        return jsonify({"name": info["name"], "count": len(df), "mean": float(values.mean()),
            "min": low, "max": high,
            "histogram": [{"label": f"{edges[i]:.1f}–{edges[i+1]:.1f}", "count": int(n)} for i, n in enumerate(frequencies)],
            "elements": [{"label": symbol, "count": n} for symbol, n in counts.most_common(15)],
            "components": [{"label": str(n), "count": count} for n, count in sorted(components.items())]})
    except (ValueError, KeyError, OSError) as exc:
        return jsonify({"error": f"无法分析数据集：{exc}"}), 400


@data_center_bp.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
        return jsonify({"success": False, "error": "Only CSV and Excel files are supported"}), 400

    if file.filename.endswith(".xlsx"):
        from importlib.util import find_spec
        if find_spec("openpyxl") is None:
            return jsonify({"success": False, "error": "Excel 导入为可选功能，请安装 openpyxl>=3.1.0；也可以将文件另存为 CSV 后上传。"}), 400

    dataset_name = request.form.get("name", "").strip() or file.filename
    if len(dataset_name) > 100:
        return jsonify({"success": False, "error": "数据集名称不能超过 100 个字符"}), 400
    timestamp = int(time.time())
    dataset_id = f"ds_{uuid.uuid4().hex}"
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1]
    save_name = f"{dataset_id}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file.save(save_path)

    try:
        df = pd.read_csv(save_path) if ext == ".csv" else pd.read_excel(save_path)
        columns = list(df.columns)

        if "formula" not in columns or "Tc" not in columns:
            os.remove(save_path)
            return jsonify({"success": False, "error": "Dataset must contain 'formula' and 'Tc' columns"}), 400

        if df.empty:
            os.remove(save_path)
            return jsonify({"success": False, "error": "数据集为空"}), 400
        problems = []
        invalid_count = 0
        for index, row in df.iterrows():
            try:
                if pd.isna(row["formula"]):
                    raise ValueError("化学式为空")
                parse_scoped_formula(row["formula"], unrestricted=True)
                tc = float(row["Tc"])
                if not math.isfinite(tc) or tc < 0:
                    raise ValueError("Tc 必须为非负有限数值")
            except (ValueError, TypeError) as exc:
                invalid_count += 1
                if len(problems) < 5:
                    problems.append(f"第 {index + 2} 行：{exc}")
        if invalid_count:
            os.remove(save_path)
            return jsonify({"success": False, "error": f"共 {invalid_count} 行不符合要求，未保存。" + "；".join(problems)}), 400
        metadata = _load_metadata()
        metadata[dataset_id] = {
            "id": dataset_id,
            "name": dataset_name,
            "file_path": os.path.relpath(save_path, config.PROJECT_ROOT),
            "rows": len(df),
            "valid_rows": len(df),
            "columns": columns,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        }
        _save_metadata(metadata)

        return jsonify({
            "success": True,
            "message": "Dataset uploaded",
            "dataset": metadata[dataset_id],
        })
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"success": False, "error": f"无法读取数据文件：{e}"}), 400


@data_center_bp.route("/<dataset_id>/preview", methods=["GET"])
def preview_dataset(dataset_id: str):
    builtin = dataset_id == BUILTIN_ID
    metadata = _load_metadata()
    if not builtin and dataset_id not in metadata:
        return jsonify({"success": False, "error": "数据集不存在"}), 404

    ds_info = _builtin_dataset() if builtin else metadata[dataset_id]
    full_path = os.path.join(config.PROJECT_ROOT, ds_info["file_path"])
    if not os.path.exists(full_path):
        return jsonify({"success": False, "error": "Dataset file missing on disk"}), 404

    try:
        page = request.args.get("page", default=1, type=int)
        if page is None or page < 1 or page > 100000:
            return jsonify({"success": False, "error": "页码无效"}), 400
        df = pd.read_csv(full_path) if full_path.endswith(".csv") else pd.read_excel(full_path)
        target = "Tc_K" if builtin else "Tc"
        # The built-in table has 81 element columns. Show the formula, temperature
        # and actual nonzero composition, not a wide, mostly empty table.
        if builtin:
            elements = [c for c in df.columns if c not in ("formula", "Tc_K")]
            selected = df.iloc[(page - 1) * 20:page * 20]
            view = selected[["formula", "Tc_K"]].copy()
            view["组成（原子分数）"] = selected[elements].apply(
                lambda row: ", ".join(f"{name}: {value:.4g}" for name, value in row.items() if value > 0), axis=1)
            description_path = os.path.join(config.PROJECT_ROOT, "data", "README.md")
            with open(description_path, encoding="utf-8") as file:
                description = file.read()
        else:
            view = df.iloc[(page - 1) * 20:page * 20]
            description = "自定义数据集，包含化学式 formula 和临界温度 Tc（K）。"
        page_size = 20
        total = len(df)
        preview_data = view.astype(object).where(pd.notna(view), None).to_dict(orient="records")
        tc_stats = {
            "min": float(df[target].min()),
            "max": float(df[target].max()),
            "mean": float(df[target].mean()),
            "count": int(df[target].count()),
        }

        return jsonify({
            "success": True,
            "preview": preview_data,
            "stats": tc_stats,
            "info": ds_info,
            "description": description,
            "page": page, "page_size": page_size, "total": total,
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Read error: {e}"}), 500


@data_center_bp.route("/<dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id: str):
    if dataset_id == BUILTIN_ID:
        return jsonify({"success": False, "error": "基础数据集不能删除"}), 403
    metadata = _load_metadata()
    if dataset_id not in metadata:
        return jsonify({"success": False, "error": "Dataset not found"}), 404

    ds_info = metadata[dataset_id]
    full_path = os.path.join(config.PROJECT_ROOT, ds_info["file_path"])
    if os.path.exists(full_path):
        os.remove(full_path)

    del metadata[dataset_id]
    _save_metadata(metadata)

    return jsonify({"success": True, "message": "Dataset deleted"})

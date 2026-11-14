"""Bounded-memory background calculations for composition-space slices."""
import base64
import csv
import io
import math
import os
import tempfile
import threading
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from app.config import config
from app.services.models import ModelService
from ml.visualization import _plot_lock
from utils.scope import METALS

JOBS = {}
LOCK = threading.RLock()
CHUNK = 4096
PREVIEW_STEPS = 1000


def validate(payload):
    elements = payload.get("elements")
    steps = payload.get("steps")
    dpi = payload.get("dpi", 300)
    fixed = payload.get("fixed_percentages", [])
    key = payload.get("model_key")
    if not isinstance(elements, list) or not 2 <= len(elements) <= 10 or any(not isinstance(e, str) for e in elements) or len(set(elements)) != len(elements):
        raise ValueError("请输入 2–10 个不同的元素")
    limit = 1000 if len(elements) == 2 else 10000
    if not isinstance(steps, int) or isinstance(steps, bool) or not 4 <= steps <= limit:
        raise ValueError(f"网格分辨率须为 4–{limit}")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or not 72 <= dpi <= 600:
        raise ValueError("DPI 须为 72–600")
    if not isinstance(fixed, list) or len(fixed) != max(0, len(elements) - 3):
        raise ValueError("请为第四种及之后的元素填写固定原子百分比")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in fixed) or sum(fixed) >= 100:
        raise ValueError("固定原子百分比须非负且总和小于 100")
    available = {item["key"] for item in ModelService.list_available(config.PROJECT_ROOT)}
    if key not in available or not ModelService.ensure_loaded(key, config.PROJECT_ROOT):
        raise ValueError("请选择可用模型")
    # Resolve by key, not active(): another request can switch the active model.
    with ModelService._lock:
        entry = ModelService._registry[key]
        model, features = entry["model"], list(entry["features"])
    if any(e not in features for e in elements):
        raise ValueError("所选元素必须包含在模型特征中")
    meta = ModelService._read_metadata_file(ModelService._model_dir(config.PROJECT_ROOT), key)
    if meta.get("composition_scope") != "custom" and any(e not in METALS for e in elements):
        raise ValueError("默认模型的相图元素须为建模范围内的金属")
    total = steps + 1 if len(elements) == 2 else (steps + 1) * (steps + 2) // 2
    return elements, steps, dpi, [x / 100 for x in fixed], key, model, features, total


def submit(payload):
    args = validate(payload)
    job_id = uuid.uuid4().hex
    with LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise ValueError("已有相图任务正在运行")
        JOBS[job_id] = {"status": "running", "completed": 0, "total": args[-1], "started_at": time.time(), "eta_seconds": None}
    threading.Thread(target=run, args=(job_id, args), daemon=True).start()
    return job_id


def run(job_id, args):
    elements, steps, dpi, fixed, key, model, features, total = args
    output_dir = os.path.join(config.RESULTS_DIR, "phase")
    path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="phase_", suffix=".csv", dir=output_dir, text=True)
        preview = []
        indices = sorted(set(round(i * steps / min(steps, PREVIEW_STEPS)) for i in range(min(steps, PREVIEW_STEPS) + 1)))
        display_grid = {(i, j) for i in indices for j in indices if i + j <= steps}
        # Ensure the triangular boundary remains closed even when rounded indices differ.
        display_grid |= {(i, steps - i) for i in indices}
        scale = 1 - sum(fixed)
        iterator = ((i, 0) for i in range(steps + 1)) if len(elements) == 2 else ((i, j) for i in range(steps + 1) for j in range(steps + 1 - i))
        minimum, maximum, done = float("inf"), float("-inf"), 0
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([*(f"{e}_atomic_fraction" for e in elements), "predicted_tc_K"])
            chunk = []
            def flush():
                nonlocal done, minimum, maximum
                if not chunk:
                    return
                values = np.asarray([item[2] for item in chunk], dtype=float)
                frame = pd.DataFrame(0.0, index=range(len(chunk)), columns=features)
                for col, element in enumerate(elements):
                    frame[element] = values[:, col]
                predictions = np.asarray(model.predict(frame), dtype=float)
                if not np.isfinite(predictions).all():
                    raise ValueError("模型返回无效预测值")
                writer.writerows([*row, float(tc)] for row, tc in zip(values, predictions))
                minimum = min(minimum, float(predictions.min()))
                maximum = max(maximum, float(predictions.max()))
                for (i, j, row), tc in zip(chunk, predictions):
                    if len(elements) == 2 or (i, j) in display_grid:
                        preview.append((i / steps, j / steps, float(tc)))
                done += len(chunk)
                elapsed = time.time() - JOBS[job_id]["started_at"]
                with LOCK:
                    JOBS[job_id].update(completed=done, eta_seconds=round(elapsed / done * (total - done)))
                chunk.clear()
            for i, j in iterator:
                row = [i / steps, 1 - i / steps] if len(elements) == 2 else [scale * i / steps, scale * j / steps, scale * (steps - i - j) / steps, *fixed]
                chunk.append((i, j, row))
                if len(chunk) >= CHUNK:
                    flush()
            flush()
        with LOCK:
            JOBS[job_id].update(stage="绘制图像", eta_seconds=None)
        image = render(elements, fixed, preview, dpi)
        with LOCK:
            JOBS[job_id].update(status="completed", image=image, path=path, model=ModelService.display_name(key, config.PROJECT_ROOT), min_tc=minimum, max_tc=maximum, elements=elements)
    except Exception as exc:
        if path and os.path.exists(path):
            os.remove(path)
        with LOCK:
            JOBS[job_id].update(status="failed", error=str(exc))


def render(elements, fixed, preview, dpi):
    data = np.asarray(preview)
    with _plot_lock:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        try:
            title = "–".join(elements)
            if fixed:
                title += " | " + ", ".join(f"{e} {v * 100:g} at.%" for e, v in zip(elements[3:], fixed))
            fig.suptitle(title, x=.05, ha="left", fontsize=11)
            if len(elements) == 2:
                ax.plot(data[:, 0] * 100, data[:, 2], color="#078b91")
                ax.set_xlabel(f"{elements[0]} (at.%)")
                ax.set_ylabel("Predicted Tc (K)")
                ax.set_xlim(0, 100)
                ax.grid(alpha=.2)
            else:
                # i=A, j=B, remainder=C; coordinates respect all three vertices.
                a, b = data[:, 0], data[:, 1]
                c = 1 - a - b
                x, y = b + .5 * c, np.sqrt(3) / 2 * c
                color = ax.tricontourf(x, y, data[:, 2], levels=20, cmap="viridis")
                draw_ternary_axes(ax, elements)
                ax.set_aspect("equal"); ax.axis("off")
                fig.colorbar(color, ax=ax, label="Predicted Tc (K)", shrink=.7)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
            return base64.b64encode(buf.getvalue()).decode("ascii")
        finally:
            plt.close(fig)


def draw_ternary_axes(ax, elements):
    """Draw each composition axis with identical outward spacing in points."""
    height = np.sqrt(3) / 2
    ax.plot([0, 1, .5, 0], [0, 0, height, 0], color="#37474f")
    # Each axis is oriented in its component's increasing direction.
    axes = [
        ((0, 0), (1, 0), (0, -1), elements[1], 0),
        ((1, 0), (.5, height), (height, .5), elements[2], -60),
        ((.5, height), (0, 0), (-height, .5), elements[0], 60),
    ]
    for start, end, outward, element, rotation in axes:
        for value in range(0, 101, 20):
            fraction = value / 100
            point = (start[0] + fraction * (end[0] - start[0]),
                     start[1] + fraction * (end[1] - start[1]))
            ax.annotate(str(value), point, xytext=(outward[0] * 10, outward[1] * 10),
                        textcoords="offset points", ha="center", va="center", fontsize=8)
        middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.annotate(f"{element} (at.%)", middle,
                    xytext=(outward[0] * 34, outward[1] * 34), textcoords="offset points",
                    ha="center", va="center", rotation=rotation, rotation_mode="anchor")

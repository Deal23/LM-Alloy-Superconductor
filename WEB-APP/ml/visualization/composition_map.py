"""Model-predicted Tc over binary/ternary composition and quaternary slices."""
import base64
import io
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.config import config
from app.services.models import ModelService
from utils.scope import ELEMENTS, METALS
from ml.visualization import _plot_lock


def composition_map(elements, key, steps, fixed, dpi):
    if not isinstance(elements, list) or not 2 <= len(elements) <= 4 or any(not isinstance(e, str) for e in elements):
        raise ValueError("请输入 2–4 个元素符号")
    if len(set(elements)) != len(elements):
        raise ValueError("元素不能重复")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 4 <= steps <= 50:
        raise ValueError("网格分辨率须为 4–50")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or not 72 <= dpi <= 600:
        raise ValueError("DPI 须为 72–600")
    if not isinstance(fixed, (int, float)) or isinstance(fixed, bool) or not math.isfinite(fixed) or not 0 <= fixed < 1:
        raise ValueError("第四元素原子分数须为 0–1（不含 1）")
    available = {item['key'] for item in ModelService.list_available(config.PROJECT_ROOT)}
    if key not in available:
        raise ValueError("请选择可用模型")
    if not ModelService.ensure_loaded(key, config.PROJECT_ROOT):
        raise ValueError("模型加载失败")
    entry = ModelService.active()
    features = entry['features']
    meta = ModelService._read_metadata_file(ModelService._model_dir(config.PROJECT_ROOT), key)
    if any(e not in features for e in elements):
        raise ValueError("所选元素必须全部包含在模型特征中")
    if meta.get('composition_scope') != 'custom' and any(e not in METALS for e in elements):
        raise ValueError("默认模型的相图元素须为建模范围内的金属")
    if any(e not in ELEMENTS for e in elements) and meta.get('composition_scope') != 'custom':
        raise ValueError("元素不在默认模型范围内")

    fractions = []
    if len(elements) == 2:
        fractions = [[i / steps, 1 - i / steps] for i in range(steps + 1)]
    else:
        scale = 1 - fixed if len(elements) == 4 else 1
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                row = [scale * i / steps, scale * j / steps, scale * (steps - i - j) / steps]
                if len(elements) == 4:
                    row.append(fixed)
                fractions.append(row)
    x = pd.DataFrame(0.0, index=range(len(fractions)), columns=features)
    for i, element in enumerate(elements):
        x[element] = [row[i] for row in fractions]
    # All grid coordinates sum to one; they are equivalent for legacy ratio
    # models, but users should interpret them as atomic fractions.
    predictions = np.asarray(entry['model'].predict(x), dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("模型返回了无效预测值")
    with _plot_lock:
        fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        try:
            if len(elements) == 2:
                vals = np.array(fractions)[:, 0]
                scatter = ax.scatter(vals, predictions, c=predictions, cmap='viridis', s=25, zorder=3)
                ax.plot(vals, predictions, color='#607d8b', linewidth=0.9)
                ax.set_xlabel(f'{elements[0]} atomic fraction')
                ax.set_ylabel('Predicted Tc (K)')
                ax.set_xlim(0, 1)
                ax.grid(alpha=.2)
            else:
                f = np.asarray(fractions)[:, :3]
                f = f / f.sum(axis=1, keepdims=True)
                xx = f[:, 1] + .5 * f[:, 2]
                yy = np.sqrt(3) / 2 * f[:, 2]
                scatter = ax.tricontourf(xx, yy, predictions, levels=20, cmap='viridis')
                ax.plot([0, 1, .5, 0], [0, 0, np.sqrt(3)/2, 0], color='#37474f', lw=1)
                ax.text(-.04, -.035, elements[0], ha='right')
                ax.text(1.04, -.035, elements[1], ha='left')
                ax.text(.5, np.sqrt(3)/2 + .035, elements[2], ha='center')
                ax.set_aspect('equal')
                ax.axis('off')
            fig.colorbar(scatter, ax=ax, label='Predicted Tc (K)', shrink=.8)
            title = '–'.join(elements)
            if len(elements) == 4:
                title += f'  |  {elements[3]} = {fixed:.1%}'
            ax.set_title(title)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
            image = base64.b64encode(buf.getvalue()).decode('ascii')
        finally:
            plt.close(fig)
    return {'success': True, 'image': image, 'elements': elements, 'model': ModelService.display_name(key, config.PROJECT_ROOT),
            'points': [{'fractions': row, 'tc': float(tc)} for row, tc in zip(fractions, predictions)],
            'min_tc': float(predictions.min()), 'max_tc': float(predictions.max())}

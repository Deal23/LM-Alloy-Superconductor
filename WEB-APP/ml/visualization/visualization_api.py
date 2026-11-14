import os
import sys
import math
import pandas as pd
from flask import Blueprint, jsonify, request, send_file

# Navigate from ml/visualization/visualization_api.py up to project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from ml.visualization import (
    generate_fig2a,
    generate_fig2b,
    generate_fig2c,
    generate_fig4b,
    generate_fig4c,
    generate_fig5,
    generate_fig5d,
    generate_ternary,
    generate_all_plots,
)

visualization_bp = Blueprint("visualization", __name__, url_prefix="/api/visualize")


@visualization_bp.route('/dataset/<plot_id>', methods=['GET'])
def dataset_plot(plot_id):
    from app.api.data_center import _builtin_dataset, _load_metadata
    from app.config import config
    generators = {'fig2a': generate_fig2a, 'fig2b': generate_fig2b, 'fig2c': generate_fig2c}
    if plot_id not in generators:
        return jsonify({'error': '图表不存在'}), 404
    dataset_id = request.args.get('dataset_id', 'builtin')
    info = _builtin_dataset() if dataset_id == 'builtin' else _load_metadata().get(dataset_id)
    if not info:
        return jsonify({'error': '数据集不存在'}), 404
    try:
        threshold = float(request.args.get('threshold', '0'))
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError('Tc 阈值须为非负数')
        path = os.path.join(config.PROJECT_ROOT, info['file_path'])
        frame = pd.read_csv(path) if path.lower().endswith('.csv') else pd.read_excel(path)
        target = 'Tc_K' if dataset_id == 'builtin' else 'Tc'
        frame[target] = pd.to_numeric(frame[target], errors='coerce')
        frame = frame[frame[target].notna() & (frame[target] >= threshold)].copy()
        if frame.empty:
            return jsonify({'error': '当前条件下没有数据'}), 400
        if target != 'Tc_K':
            frame['Tc_K'] = frame[target]
        result = generators[plot_id](dataframe=frame, tc_threshold=threshold, dpi=request.args.get('dpi', type=int, default=300))
        if not result.get('success'):
            return jsonify({'error': result.get('error', '图表生成失败')}), 400
        return jsonify({'success': True, 'image': result['image'], 'dataset': info['name'], 'count': len(frame), 'threshold': threshold})
    except (ValueError, KeyError, OSError) as exc:
        return jsonify({'error': f'无法生成图表：{exc}'}), 400


@visualization_bp.route('/phase/jobs', methods=['POST'])
def start_phase_job():
    from ml.visualization.phase_jobs import submit
    try:
        job_id = submit(request.get_json(silent=True) or {})
        return jsonify({'job_id': job_id}), 202
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@visualization_bp.route('/phase/jobs/<job_id>', methods=['GET'])
def phase_job_status(job_id):
    from ml.visualization.phase_jobs import JOBS, LOCK
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': '任务不存在，服务可能已重启'}), 404
        return jsonify({k: v for k, v in job.items() if k != 'path'})


@visualization_bp.route('/phase/jobs/<job_id>/download', methods=['GET'])
def phase_job_download(job_id):
    from ml.visualization.phase_jobs import JOBS, LOCK
    with LOCK:
        job = JOBS.get(job_id)
        if not job or job['status'] != 'completed':
            return jsonify({'error': '结果尚不可用'}), 404
        path = job['path']
    return send_file(path, mimetype='text/csv', as_attachment=True, download_name='LMSC_composition_predictions.csv')


@visualization_bp.route('/composition', methods=['POST'])
def api_composition():
    from ml.visualization.composition_map import composition_map
    payload = request.get_json(silent=True) or {}
    try:
        result = composition_map(payload.get('elements'), payload.get('model_key'),
                                 payload.get('steps', 20), payload.get('fixed_fraction', 0.1), payload.get('dpi', 300))
        return jsonify(result)
    except (ValueError, TypeError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400


@visualization_bp.before_request
def validate_dpi():
    if "dpi" not in request.args:
        return None
    try:
        dpi = int(request.args["dpi"])
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "DPI 必须是 72–600 的整数"}), 400
    if not 72 <= dpi <= 600:
        return jsonify({"success": False, "error": "DPI 必须是 72–600 的整数"}), 400


@visualization_bp.route("/fig2a", methods=["GET"])
def api_fig2a():
    tc_threshold = request.args.get("tc_threshold", type=float, default=0)
    dpi = request.args.get("dpi", type=int, default=300)
    result = generate_fig2a(tc_threshold=tc_threshold, dpi=dpi, return_base64=True)
    if result["success"]:
        return jsonify(
            {
                "success": True,
                "image": result["image"],
                "data_info": result.get("data_info", {}),
            }
        )
    return jsonify({"success": False, "error": result["error"]}), 400


@visualization_bp.route("/fig2b", methods=["GET"])
def api_fig2b():
    dpi = request.args.get("dpi", type=int, default=300)
    result = generate_fig2b(dpi=dpi, return_base64=True)
    if result["success"]:
        return jsonify(
            {
                "success": True,
                "image": result["image"],
                "data_info": result.get("data_info", {}),
            }
        )
    return jsonify({"success": False, "error": result["error"]}), 400


@visualization_bp.route("/fig2c", methods=["GET"])
def api_fig2c():
    tc_threshold = request.args.get("tc_threshold", type=float, default=5)
    dpi = request.args.get("dpi", type=int, default=300)
    result = generate_fig2c(tc_threshold=tc_threshold, dpi=dpi, return_base64=True)
    if result["success"]:
        return jsonify(
            {
                "success": True,
                "image": result["image"],
                "data_info": result.get("data_info", {}),
            }
        )
    return jsonify({"success": False, "error": result["error"]}), 400


@visualization_bp.route("/fig4b", methods=["GET"])
def api_fig4b():
    dpi = request.args.get("dpi", type=int, default=300)
    result = generate_fig4b(dpi=dpi, return_base64=True)
    if result["success"]:
        return jsonify({"success": True, "image": result["image"]})
    return jsonify({"success": False, "error": result["error"]}), 400


@visualization_bp.route("/fig4c", methods=["GET"])
def api_fig4c():
    model_type = request.args.get("model_type", type=str, default="ET")
    dpi = request.args.get("dpi", type=int, default=300)
    result = generate_fig4c(model_type=model_type, dpi=dpi, return_base64=True)
    if result["success"]:
        return jsonify(
            {
                "success": True,
                "image": result["image"],
                "data_info": result.get("data_info", {}),
            }
        )
    return jsonify({"success": False, "error": result["error"]}), 400


@visualization_bp.route("/fig5", methods=["GET"])
def api_fig5():
    model_type = request.args.get("model_type", type=str, default="ET")
    dpi = request.args.get("dpi", type=int, default=300)
    result = generate_fig5(model_type=model_type, dpi=dpi, return_base64=True)
    if result["success"]:
        return jsonify(
            {
                "success": True,
                "image": result["image"],
                "metrics": result.get("metrics", {}),
            }
        )
    return jsonify({"success": False, "error": result["error"]}), 400


@visualization_bp.route("/fig5d", methods=["GET"])
def api_fig5d():
    dpi = request.args.get("dpi", type=int, default=300)
    result = generate_fig5d(dpi=dpi, return_base64=True)
    if result["success"]:
        return jsonify({"success": True, "image": result["image"]})
    return jsonify({"success": False, "error": result["error"]}), 400


@visualization_bp.route("/ternary", methods=["GET"])
def api_ternary():
    model_type = request.args.get("model_type", type=str, default="ET")
    elements_str = request.args.get("elements", type=str, default=None)
    dpi = request.args.get("dpi", type=int, default=300)

    elements = None
    if elements_str:
        try:
            elements = elements_str.split(",")
            if len(elements) != 3:
                return jsonify(
                    {
                        "success": False,
                        "error": "elements must be exactly 3 elements separated by comma",
                    }
                ), 400
        except Exception:
            return jsonify({"success": False, "error": "Invalid elements format"}), 400

    result = generate_ternary(
        model_type=model_type, elements=elements, dpi=dpi, return_base64=True
    )
    if result["success"]:
        return jsonify(
            {
                "success": True,
                "image": result["image"],
                "data_info": result.get("data_info", {}),
            }
        )
    return jsonify({"success": False, "error": result["error"]}), 400


@visualization_bp.route("/all", methods=["GET"])
def api_all():
    dpi = request.args.get("dpi", type=int, default=300)
    results = generate_all_plots(dpi=dpi)

    response = {}
    for plot_name, result in results.items():
        if result["success"]:
            response[plot_name] = {
                "success": True,
                "image": result.get("image", ""),
                "data_info": result.get("data_info", {}),
                "metrics": result.get("metrics", {}),
            }
        else:
            response[plot_name] = {
                "success": False,
                "error": result.get("error", "Unknown error"),
            }

    return jsonify(response)


@visualization_bp.route("/list", methods=["GET"])
def api_list():
    return jsonify(
        {
            "plots": [
                {
                    "id": "fig2a",
                    "name": "Figure 2a - Tc Distribution",
                    "description": "Tc distribution histogram with KDE",
                },
                {
                    "id": "fig2b",
                    "name": "Figure 2b - Element Count",
                    "description": "Number of elements per formula",
                },
                {
                    "id": "fig2c",
                    "name": "Figure 2c - Periodic Table",
                    "description": "Element frequency heatmap",
                },
                {
                    "id": "fig4b",
                    "name": "Figure 4b - Convergence",
                    "description": "Prediction convergence with data size",
                },
                {
                    "id": "fig4c",
                    "name": "Figure 4c - Binary Alloy Heatmap",
                    "description": "Binary alloy Tc heatmap",
                },
                {
                    "id": "fig5",
                    "name": "Figure 5 - Model Performance",
                    "description": "Prediction vs experiment scatter plot",
                },
                {
                    "id": "fig5d",
                    "name": "Figure 5d - Model Comparison",
                    "description": "RF vs ET vs Experiment comparison",
                },
                {
                    "id": "ternary",
                    "name": "Ternary Phase Diagram",
                    "description": "Ga-In-Sn ternary contour plot",
                },
            ]
        }
    )

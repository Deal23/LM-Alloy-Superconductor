"""Lightweight page and API integration checks without loading local models."""

from html.parser import HTMLParser
from pathlib import Path

import pytest

from app import create_app
from app.services.models import ModelService


def test_documented_optional_requirements_and_docs_exist():
    root = Path(__file__).resolve().parents[1]
    for path in ("requirements.txt", "docs/CLEANUP.md", "docs/FILE_LOCATIONS.md"):
        assert (root / path).is_file(), path


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "script" and "src" in attrs:
            self.scripts.append(attrs["src"])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ModelService, "active", lambda: None)
    monkeypatch.setattr(ModelService, "ensure_loaded", lambda *args: None)
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_workspace_renders_without_models(client):
    response = client.get("/")
    assert response.status_code == 200
    parser = PageParser()
    parser.feed(response.get_data(as_text=True))
    assert len(parser.ids) == len(set(parser.ids))
    assert {"dash-formula", "btn-dash-predict", "model-notice", "batch-count",
            "theme-toggle", "image-dialog", "sidebar-backdrop", "viz-elements"} <= set(parser.ids)
    assert parser.scripts == ["/static/app.js?v=23"]


def test_resources_endpoint(client):
    response = client.get("/api/resources")
    assert response.status_code == 200
    assert response.json["logical_cpus"] >= 1
    assert response.json["memory_bytes"] is None or response.json["memory_bytes"] > 0


def test_builtin_dataset_count_matches_preview(client):
    dataset = client.get("/api/dataset/list").json["datasets"][0]
    preview = client.get("/api/dataset/builtin/preview").json
    assert dataset["rows"] == preview["total"]
    assert dataset["valid_rows"] == preview["stats"]["count"]


def test_task_list_only_exposes_supported_screening(client):
    response = client.get("/api/tasks")
    assert response.status_code == 200
    assert set(response.json["tasks"]) == {"alloy_predict"}


def test_screening_progress_events(tmp_path):
    import time
    from app.services.tasks import TaskService
    script = tmp_path / "fake_task.py"
    script.write_text('print(\'LMSC_PROGRESS {"completed": 1, "total": 2}\', flush=True)\nprint(\'LMSC_PROGRESS {"completed": 2, "total": 2}\', flush=True)\n')
    job_id = "progress-test"
    with TaskService._lock:
        TaskService._status[job_id] = "queued"
        TaskService._progress[job_id] = {"started_at": time.time(), "completed": 0, "total": None, "eta_seconds": None}
    TaskService._run(str(tmp_path), str(script), job_id, {})
    result = TaskService.status(job_id)
    assert result["status"].startswith("completed")
    assert result["progress"]["completed"] == 2
    assert result["progress"]["total"] == 2
    assert result["progress"]["eta_seconds"] == 0


def test_screening_result_columns_are_initialized_without_warnings():
    import ast
    import warnings
    import numpy as np
    import pandas as pd

    script = Path(__file__).resolve().parents[1] / "ml/screening/alloy_predict.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ensure_result_columns")
    namespace = {"pd": pd, "np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(script), "exec"), namespace)
    frame = pd.DataFrame({"elements": ["Ga-In"], "calculated": ["no"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        namespace["ensure_result_columns"](frame)
        frame.at[0, "Predicted_Tc"] = 5.0
        frame.at[0, "Elements"] = "Ga0.5In0.5"
        frame.at[0, "stability_variance"] = 0.1
        frame.at[0, "stability_log"] = "[5.0]"
    assert frame.at[0, "Elements"] == "Ga0.5In0.5"


def test_composition_maps_use_selected_model(client, monkeypatch):
    import numpy as np
    import pytest
    pytest.importorskip("matplotlib")

    class Model:
        def predict(self, frame):
            assert np.allclose(frame.sum(axis=1), 1)
            return frame["Ga"].to_numpy() * 10 + frame["In"].to_numpy() * 5

    entry = {"key": "ET_test", "features": ["Ga", "In", "Sn", "Bi"], "model": Model(), "metrics": {}}
    monkeypatch.setattr(ModelService, "list_available", lambda root: [{"key": "ET_test"}])
    monkeypatch.setattr(ModelService, "ensure_loaded", lambda *args: "ET_test")
    monkeypatch.setattr(ModelService, "active", lambda: entry)
    for elements, expected in [(["Ga", "In"], 5), (["Ga", "In", "Sn"], 15), (["Ga", "In", "Sn", "Bi"], 15)]:
        response = client.post('/api/visualize/composition', json={"elements": elements, "model_key": "ET_test", "steps": 4, "dpi": 72, "fixed_fraction": .2})
        assert response.status_code == 200, response.json
        assert len(response.json['points']) == expected
        assert response.json['image']
        if len(elements) == 4:
            assert all(p['fractions'][3] == .2 for p in response.json['points'])
    response = client.post('/api/visualize/composition', json={"elements": ["Ga", "Ga"], "model_key": "ET_test"})
    assert response.status_code == 400


def test_phase_job_slices_and_streams_results(client, monkeypatch, tmp_path):
    import csv
    import time
    import numpy as np
    from ml.visualization import phase_jobs

    class Model:
        def predict(self, frame):
            assert np.allclose(frame.sum(axis=1), 1)
            return frame['Ga'].to_numpy() * 10 + frame['In'].to_numpy() * 5

    key = 'ET_slice'
    monkeypatch.setattr(ModelService, '_registry', {key: {'model': Model(), 'features': ['Ga', 'In', 'Sn', 'Bi', 'Zn', 'Ag']}})
    monkeypatch.setattr(ModelService, 'list_available', lambda root: [{'key': key}])
    monkeypatch.setattr(ModelService, 'ensure_loaded', lambda *args: key)
    monkeypatch.setattr(phase_jobs.config, 'RESULTS_DIR', str(tmp_path))
    monkeypatch.setattr(phase_jobs, 'JOBS', {})
    payload = {'elements': ['Ga', 'In', 'Sn', 'Bi', 'Zn', 'Ag'], 'steps': 4, 'dpi': 72, 'fixed_percentages': [10, 5, 2], 'model_key': key}
    response = client.post('/api/visualize/phase/jobs', json=payload)
    assert response.status_code == 202
    job = response.json['job_id']
    for _ in range(200):
        state = client.get(f'/api/visualize/phase/jobs/{job}').json
        if state['status'] != 'running':
            break
        time.sleep(.02)
    assert state['status'] == 'completed', state
    assert state['completed'] == state['total'] == 15
    result = client.get(f'/api/visualize/phase/jobs/{job}/download')
    rows = list(csv.reader(result.get_data(as_text=True).splitlines()))
    assert len(rows) == 16
    assert [float(v) for v in rows[1][3:6]] == [.1, .05, .02]
    assert client.post('/api/visualize/phase/jobs', json={**payload, 'fixed_percentages': [90, 10, 0]}).status_code == 400
    args = phase_jobs.validate({**payload, 'steps': 10000})
    assert args[-1] == 50015001


def test_ternary_axis_labels_use_equal_offsets():
    import math
    import matplotlib.pyplot as plt
    import pytest
    from ml.visualization.phase_jobs import draw_ternary_axes

    fig, ax = plt.subplots()
    try:
        draw_ternary_axes(ax, ['Ga', 'In', 'Sn'])
        labels = [text for text in ax.texts if '(at.%)' in text.get_text()]
        assert [text.get_text() for text in labels] == ['In (at.%)', 'Sn (at.%)', 'Ga (at.%)']
        assert all(math.hypot(*text.get_position()) == pytest.approx(34) for text in labels)
        ticks = [text for text in ax.texts if text.get_text().isdigit()]
        assert len(ticks) == 18
        assert all(math.hypot(*text.get_position()) == pytest.approx(10) for text in ticks)
    finally:
        plt.close(fig)


def test_ternary_axes_follow_cyclic_increasing_order():
    import matplotlib.pyplot as plt
    import pytest
    from ml.visualization.phase_jobs import draw_ternary_axes
    fig, ax = plt.subplots()
    try:
        draw_ternary_axes(ax, ['Ga', 'In', 'Sn'])
        ticks = [text for text in ax.texts if text.get_text().isdigit()]
        # Bottom: left to right, right side: bottom to top, left side: top to bottom.
        for index in range(3):
            assert [t.get_text() for t in ticks[index * 6:(index + 1) * 6]] == ['0', '20', '40', '60', '80', '100']
        assert ticks[0].xy == pytest.approx((0, 0))
        assert ticks[5].xy == pytest.approx((1, 0))
        assert ticks[6].xy == pytest.approx((1, 0))
        assert ticks[11].xy == pytest.approx((.5, 3**.5 / 2))
        assert ticks[12].xy == pytest.approx((.5, 3**.5 / 2))
        assert ticks[17].xy == pytest.approx((0, 0))
    finally:
        plt.close(fig)


def test_builtin_analysis_works_without_optional_visualization(client):
    response = client.get("/api/dataset/builtin/analysis?threshold=5")
    assert response.status_code == 200
    data = response.json
    assert 0 < data["count"] < 12290
    assert sum(item["count"] for item in data["histogram"]) == data["count"]
    assert sum(item["count"] for item in data["components"]) == data["count"]
    assert "O" not in [item["label"] for item in data["elements"]]
    assert client.get("/api/dataset/builtin/analysis?threshold=-1").status_code == 400
    assert client.get("/api/dataset/absent/analysis").status_code == 404


def test_dataset_png_uses_selected_dataset_and_threshold(client, monkeypatch, tmp_path):
    from app.api import data_center
    from ml.visualization import visualization_api

    path = tmp_path / 'custom.csv'
    path.write_text('formula,Tc\nH2,1\nYBa2Cu3O7,90\nMgB2,39\n', encoding='utf-8')
    monkeypatch.setattr(data_center, '_load_metadata', lambda: {'custom': {'name': '我的数据', 'file_path': str(path)}})
    observed = []
    def generate(**kwargs):
        observed.append(kwargs)
        return {'success': True, 'image': 'image'}
    monkeypatch.setattr(visualization_api, 'generate_fig2a', generate)
    response = client.get('/api/visualize/dataset/fig2a?dataset_id=custom&threshold=39&dpi=300')
    assert response.status_code == 200
    assert response.json['dataset'] == '我的数据'
    assert response.json['count'] == 2
    assert observed[0]['dataframe']['Tc_K'].tolist() == [90, 39]
    assert observed[0]['dpi'] == 300
    assert client.get('/api/visualize/dataset/fig2a?dataset_id=custom&threshold=100').status_code == 400
    assert client.get('/api/visualize/dataset/fig2a?dataset_id=missing').status_code == 404


def test_builtin_dataset_details_are_read_only_and_paginated(client):
    listing = client.get("/api/dataset/list").json
    builtin = next(item for item in listing["datasets"] if item.get("builtin"))
    assert builtin["id"] == "builtin"
    first = client.get("/api/dataset/builtin/preview?page=1").json
    second = client.get("/api/dataset/builtin/preview?page=2").json
    assert first["total"] == 12290
    assert len(first["preview"]) == len(second["preview"]) == 20
    assert first["preview"] != second["preview"]
    assert "Tc_K" in first["preview"][0]
    assert "组成（原子分数）" in first["preview"][0]
    assert "元素锝" in first["description"]
    assert client.delete("/api/dataset/builtin").status_code == 403
    assert client.get("/api/dataset/builtin/preview?page=0").status_code == 400


@pytest.mark.parametrize("path", ["/static/app.js", "/static/style.css", "/workflow"])
def test_workspace_assets_and_legacy_route(client, path):
    assert client.get(path).status_code == 200


def test_prediction_contract_still_accepts_selected_model(client, monkeypatch):
    calls = []

    def predict(formula, key, root):
        calls.append((formula, key))
        return {"formula": formula, "model": key, "predicted_tc": 38.5}

    monkeypatch.setattr(ModelService, "predict_formula", predict)
    response = client.post("/api/predict", json={"formula": " MgB2 ", "model_key": "ET_custom"})
    assert response.json["predicted_tc"] == 38.5
    assert calls == [("MgB2", "ET_custom")]
    response = client.post("/api/batch_predict", json={"formulas": ["MgB2", "Nb"], "model_key": "ET_custom"})
    assert len(response.json["results"]) == 2
    assert calls[-2:] == [("MgB2", "ET_custom"), ("Nb", "ET_custom")]


def test_prediction_with_real_tree_model(client, monkeypatch):
    """Use an ephemeral model; never train over or save to the user's model library."""
    import pandas as pd
    from sklearn.ensemble import ExtraTreesRegressor

    features = ["Mg", "B"]
    model = ExtraTreesRegressor(n_estimators=3, random_state=42)
    model.fit(pd.DataFrame([[1, 2], [2, 1], [1, 1]], columns=features), [39, 10, 20])
    monkeypatch.setattr(ModelService, "_registry", {
        "ET_test": {"model": model, "features": features, "metrics": {"R2": 0.9, "RMSE": 2.0}}
    })
    monkeypatch.setattr(ModelService, "_training_proximity", lambda *args: 0.8)
    response = client.post("/api/predict", json={"formula": "MgB2", "model_key": "ET_test"})
    assert response.status_code == 200
    assert response.json["formula"] == "MgB2"
    assert isinstance(response.json["predicted_tc"], (int, float))
    assert 0 <= response.json["reliability"]["score"] <= 100


def test_background_training_progress_and_name(client, monkeypatch):
    import time
    from app.api import core

    def train(**kwargs):
        kwargs["progress"]("训练决策树 10/10", 85)
        return True, {"name": kwargs["display_name"], "model_key": "ET_test"}

    monkeypatch.setattr(core.TrainingService, "train", train)
    monkeypatch.setattr(core, "_training_jobs", {})
    response = client.post("/api/training_jobs", json={"n_estimators": 10, "display_name": "我的模型"})
    assert response.status_code == 202
    job_id = response.json["job_id"]
    for _ in range(100):
        result = client.get(f"/api/training_jobs/{job_id}").json
        if result["status"] != "running":
            break
        time.sleep(0.01)
    assert result["status"] == "completed"
    assert result["progress"] == 100
    assert result["result"]["name"] == "我的模型"
    assert client.post("/api/training_jobs", json={"n_estimators": 1}).status_code == 400


def test_screening_rejects_out_of_scope_elements(client):
    response = client.post("/api/run_task", json={"task_name": "alloy_predict", "params": {"elements": ["Ga", "O"], "num_elements": 2}})
    assert response.status_code == 400

import io
import json

from app.api import data_center
from app.api.core import _resolve_dataset


def test_upload_index_is_in_data(tmp_path, monkeypatch):
    current = tmp_path / "data/uploads/datasets.json"
    entries = {"new": {"file_path": "data/uploads/new.csv"}}
    monkeypatch.setattr(data_center, "METADATA_FILE", str(current))
    assert data_center._load_metadata() == {}
    data_center._save_metadata(entries)
    assert current.exists()
    assert data_center._load_metadata() == entries
    assert _resolve_dataset("new").endswith("data/uploads/new.csv")


def test_web_plot_preview_does_not_export_files(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("matplotlib")
    from ml import visualization

    def plot():
        visualization.plt.figure()
        visualization.plt.plot([0, 1], [1, 2])
        visualization.plt.savefig(tmp_path / "unwanted.tif", dpi=600)

    original = visualization.plt.savefig
    result = visualization._generate_plot(plot, dpi=72)
    assert result["success"]
    assert result["image"]
    assert not list(tmp_path.iterdir())
    assert visualization.plt.savefig is original


def test_paper_plots_render_with_matplotlib():
    import pytest
    pytest.importorskip("matplotlib")
    from ml.visualization import generate_fig2a, generate_fig2b, generate_fig2c
    for generator in (generate_fig2a, generate_fig2b, generate_fig2c):
        result = generator(dpi=72)
        assert result["success"], result.get("error")
        assert result["image"]


def test_histogram_axes_follow_data(tmp_path, monkeypatch):
    import pandas as pd
    import pytest
    pytest.importorskip("matplotlib")
    from ml.visualization import plot_generator as plots

    pd.DataFrame({"Tc_K": [0, 200, 250, 300]}).to_csv(tmp_path / "superconductors.csv", index=False)
    monkeypatch.setattr(plots, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(plots.plt, "show", lambda: None)
    monkeypatch.setattr(plots.plt, "savefig", lambda *args, **kwargs: None)
    try:
        plots.plot_fig2a()
        ax, density = plots.plt.gcf().axes
        assert ax.get_xlim()[1] >= 300
        assert ax.get_ylim()[1] >= 1
        assert density.get_ylim()[1] > 0
    finally:
        plots.plt.close("all")


def test_visualization_dpi_controls_png_size(monkeypatch):
    import base64
    from io import BytesIO
    import pytest
    from app import create_app
    from app.services.models import ModelService
    monkeypatch.setattr(ModelService, "ensure_loaded", lambda *args: None)
    client = create_app().test_client()
    image_module = pytest.importorskip("PIL.Image")
    images = []
    for query in ("", "?dpi=150", "?dpi=300"):
        response = client.get("/api/visualize/fig2b" + query)
        assert response.status_code == 200
        images.append(image_module.open(BytesIO(base64.b64decode(response.json["image"]))))
    default, low, high = images
    assert default.size == high.size
    assert high.width > low.width * 1.9
    assert high.height > low.height * 1.9
    assert high.info["dpi"][0] == pytest.approx(300, abs=1)
    for invalid in ("0", "601", "nan", "300.5"):
        assert client.get(f"/api/visualize/fig2b?dpi={invalid}").status_code == 400


def test_csv_upload_uses_runtime_folder_without_excel(tmp_path, monkeypatch):
    from app import create_app
    from app.services.models import ModelService

    monkeypatch.setattr(ModelService, "ensure_loaded", lambda *args: None)
    monkeypatch.setattr(data_center, "UPLOAD_DIR", str(tmp_path / "data/uploads"))
    monkeypatch.setattr(data_center, "METADATA_FILE", str(tmp_path / "data/uploads/datasets.json"))
    client = create_app().test_client()
    response = client.post('/api/dataset/upload', data={
        'file': (io.BytesIO(b'formula,Tc\nMgB2,39\n'), 'sample.csv')
    })
    assert response.status_code == 200
    entry = response.json['dataset']
    assert (tmp_path / 'data/uploads/datasets.json').exists()
    assert entry['file_path'].endswith('.csv')
    assert len(list((tmp_path / 'data/uploads').iterdir())) == 2


def test_invalid_upload_is_not_saved(tmp_path, monkeypatch):
    from app import create_app
    monkeypatch.setattr(data_center, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(data_center, "METADATA_FILE", str(tmp_path / "uploads/datasets.json"))
    client = create_app().test_client()
    response = client.post('/api/dataset/upload', data={
        'file': (io.BytesIO(b'formula,Tc\nMgB2,39\nInvalid!,90\n'), 'bad.csv')
    })
    assert response.status_code == 400
    assert '第 3 行' in response.json['error']
    assert not list((tmp_path / 'uploads').iterdir())


def test_custom_upload_accepts_oxygen_hydrogen_and_nonmetal(tmp_path, monkeypatch):
    from app import create_app
    monkeypatch.setattr(data_center, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(data_center, "METADATA_FILE", str(tmp_path / "uploads/datasets.json"))
    client = create_app().test_client()
    response = client.post('/api/dataset/upload', data={
        'file': (io.BytesIO(b'formula,Tc\nYBa2Cu3O7,90\nH2,0\nB2C,1\n'), 'custom.csv')
    })
    assert response.status_code == 200
    assert response.json['dataset']['valid_rows'] == 3

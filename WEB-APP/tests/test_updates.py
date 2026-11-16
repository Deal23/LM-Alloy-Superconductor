import io
import base64
import json
import zipfile

import pytest

from app.services import updates


def package(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, content in files.items():
            archive.writestr("repository-commit/WEB-APP/" + name, content)
    return out.getvalue()


def mock_release(monkeypatch, content):
    monkeypatch.setattr(updates, "latest_release", lambda: {
        "newer": True, "ready": True, "latest": "0.1.1", "commit": "a" * 40,
    })
    monkeypatch.setattr(updates, "_get", lambda url, limit: content)


def test_update_preserves_user_data_and_removes_stale_program_files(tmp_path, monkeypatch):
    (tmp_path / "app").mkdir()
    (tmp_path / "app/__init__.py").write_text("old")
    (tmp_path / "app/obsolete.py").write_text("obsolete")
    (tmp_path / "data/uploads").mkdir(parents=True)
    (tmp_path / "data/uploads/data.csv").write_text("private")
    (tmp_path / "ml/models").mkdir(parents=True)
    (tmp_path / "ml/models/model.joblib").write_text("model")
    (tmp_path / "app/.local").write_text("private")
    content = package({"app/__init__.py": '__version__ = "0.1.1"', "run.py": "run", "requirements.txt": "requirements"})
    mock_release(monkeypatch, content)
    result = updates.apply_update(tmp_path, "0.1.1")
    assert (tmp_path / "app/__init__.py").read_text() == '__version__ = "0.1.1"'
    assert result["version"] == "0.1.1"
    assert not (tmp_path / "app/obsolete.py").exists()
    assert not (tmp_path / ".update-backups").exists()
    assert (tmp_path / "data/uploads/data.csv").read_text() == "private"
    assert (tmp_path / "ml/models/model.joblib").read_text() == "model"
    assert (tmp_path / "app/.local").read_text() == "private"


def test_update_failure_restores_overwritten_files(tmp_path, monkeypatch):
    (tmp_path / "app").mkdir()
    (tmp_path / "app/__init__.py").write_text("old")
    content = package({"app/__init__.py": '__version__ = "0.1.1"',
                       "run.py": "run", "requirements.txt": "requirements"})
    mock_release(monkeypatch, content)
    original = updates.shutil.copyfile

    def fail_on_run(src, dst, *args, **kwargs):
        if str(dst).endswith("run.py"):
            raise OSError("simulated failure")
        return original(src, dst, *args, **kwargs)

    monkeypatch.setattr(updates.shutil, "copyfile", fail_on_run)
    with pytest.raises(OSError, match="simulated failure"):
        updates.apply_update(tmp_path, "0.1.1")
    assert (tmp_path / "app/__init__.py").read_text() == "old"
    assert not (tmp_path / "requirements.txt").exists()


@pytest.mark.parametrize("name", ["../escape", "app/../../x"])
def test_update_rejects_unsafe_files(tmp_path, monkeypatch, name):
    content = package({"app/__init__.py": "new", "run.py": "run", "requirements.txt": "req", name: "bad"})
    mock_release(monkeypatch, content)
    with pytest.raises(ValueError):
        updates.apply_update(tmp_path, "0.1.1")
    assert not (tmp_path / "run.py").exists()


def test_default_branch_version_uses_pinned_commit(monkeypatch):
    monkeypatch.setattr(updates, "__version__", "0.1.0")
    def get(url, limit):
        if "/commits?" in url:
            return json.dumps([{"sha": "b" * 40}]).encode()
        assert "b" * 40 in url
        return json.dumps({"encoding": "base64", "content": base64.b64encode(
            b'__version__ = "0.1.2"').decode()}).encode()
    monkeypatch.setattr(updates, "_get", get)
    info = updates.latest_release()
    assert info["latest"] == "0.1.2"
    assert info["newer"] and info["ready"]
    assert info["commit"] == "b" * 40


def test_repository_archive_preserves_local_data(tmp_path, monkeypatch):
    content = package({"app/__init__.py": '__version__ = "0.1.1"', "run.py": "run",
                       "requirements.txt": "req", "data/uploads/x": "remote",
                       "ml/models/model.joblib": "remote"})
    mock_release(monkeypatch, content)
    updates.apply_update(tmp_path, "0.1.1")
    assert not (tmp_path / "data/uploads/x").exists()
    assert not (tmp_path / "ml/models/model.joblib").exists()


def test_remote_and_cross_origin_update_denied(monkeypatch):
    from app import create_app
    from app.services.models import ModelService
    monkeypatch.setattr(ModelService, "ensure_loaded", lambda *args: None)
    client = create_app().test_client()
    assert client.get('/api/updates', environ_base={'REMOTE_ADDR': '192.0.2.1'}).status_code == 403
    assert client.post('/api/updates/apply', json={'version': '0.1.1'}, headers={'Origin': 'https://evil.example'}).status_code == 403

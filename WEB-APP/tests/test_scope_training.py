import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from app.services.models import ModelService
from app.services.training import TrainingService
from utils.scope import parse_scoped_formula


@pytest.mark.parametrize("formula", ["YBa2Cu3O7", "H3S", "B2C", "MgB2!", "Mg0B2"])
def test_out_of_scope_formulas_rejected(formula):
    with pytest.raises(ValueError):
        parse_scoped_formula(formula)


def test_valid_formulas_are_normalized():
    assert parse_scoped_formula("MgB2") == {"Mg": 1 / 3, "B": 2 / 3}
    assert parse_scoped_formula("Nb3Sn") == {"Nb": 0.75, "Sn": 0.25}


def test_frozen_base_dataset_checksum_and_schema():
    path = Path(__file__).resolve().parents[1] / "data/superconductors.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "7ea2fede6f0b3e633f5d388ef35a4bf7ff326898ea670cf58e1271a3b7e076fb"
    df = pd.read_csv(path)
    assert len(df) == 12290
    assert "Tc_K" in df.columns and "Tc" in df.columns


def test_legacy_evaluation_plots_are_not_simulated():
    pytest.importorskip("matplotlib")
    from ml.visualization import generate_fig5, generate_fig5d, generate_fig4b, generate_fig4c
    for generator in (generate_fig5, generate_fig5d, generate_fig4b, generate_fig4c):
        assert generator()["success"] is False


def test_latest_dataset_training_uses_tc_k_and_atomic_fractions(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    path = root / "data/superconductors.csv"
    assert path.exists()
    monkeypatch.setattr(ModelService, "_registry", {})
    monkeypatch.setattr(ModelService, "_active_key", None)
    updates = []
    success, result = TrainingService.train(
        project_root=str(tmp_path), data_path=str(path), n_estimators=10,
        display_name="测试模型", progress=lambda stage, value: updates.append((stage, value)),
    )
    assert success, result
    assert result["name"] == "测试模型"
    assert [value for _, value in updates] == sorted(value for _, value in updates)
    metadata = json.loads((tmp_path / "ml/models" / (result["model_key"] + "_metadata.json")).read_text())
    assert metadata["feature_encoding"] == "atomic_fraction"
    assert metadata["dataset"]["path"] == "superconductors.csv"
    assert result["n_samples_train"] + result["n_samples_test"] == len(pd.read_csv(path)) == 12290
    prediction = ModelService.predict_formula("MgB2", result["model_key"], str(tmp_path))
    assert "predicted_tc" in prediction
    assert "error" in ModelService.predict_formula("YBa2Cu3O7", result["model_key"], str(tmp_path))


def test_custom_dataset_allows_oxygen_and_nonmetal(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelService, "_registry", {})
    monkeypatch.setattr(ModelService, "_active_key", None)
    path = tmp_path / "custom.csv"
    path.write_text("formula,Tc\nMgB2,39\nYBa2Cu3O7,90\nH2,0\nB2C,1\nNb,9\nSn,3\n", encoding="utf-8")
    success, result = TrainingService.train(project_root=str(tmp_path), data_path=str(path), n_estimators=10)
    assert success, result
    prediction = ModelService.predict_formula("YBa2Cu3O7", result["model_key"], str(tmp_path))
    assert "predicted_tc" in prediction
    assert prediction["reliability"]["details"]["training_proximity"] is None
    assert "predicted_tc" in ModelService.predict_formula("H2", result["model_key"], str(tmp_path))
    assert "error" in ModelService.predict_formula("He", result["model_key"], str(tmp_path))

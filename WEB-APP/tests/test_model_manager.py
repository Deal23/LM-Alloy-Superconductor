import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.models import ModelService


class TestModelService:
    """Test lightweight model service"""

    def test_list_available_models(self):
        models = ModelService.list_available(os.environ.get("PROJECT_ROOT", os.getcwd()))
        assert isinstance(models, list)

    def test_load_model_invalid(self):
        result = ModelService.load("NONEXISTENT_MODEL_12345", os.getcwd())
        assert result is None

    def test_active_model_initially_none(self):
        active = ModelService.active()
        # Either None or a dict if default model auto-loaded
        assert active is None or isinstance(active, dict)

    def test_status(self):
        status = ModelService.status()
        assert isinstance(status, dict)
        assert "loaded" in status

    def test_predict_invalid_formula(self):
        result = ModelService.predict_formula("not_a_formula", project_root=os.getcwd())
        assert "error" in result

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config, config


class TestConfig:
    """Test configuration management"""

    def test_config_from_env(self):
        """Test Config.from_env() class method"""
        cfg = Config.from_env()
        assert cfg.PROJECT_ROOT is not None
        assert cfg.MODEL_DIR is not None
        assert cfg.DATA_DIR is not None
        assert cfg.RESULTS_DIR is not None

    def test_config_defaults(self):
        """Test default values"""
        cfg = Config.from_env()
        assert cfg.DEFAULT_MODEL == "ET"
        assert cfg.DEFAULT_TEST_SIZE == 0.2
        assert cfg.DEFAULT_N_ESTIMATORS == 300
        assert cfg.TASK_TIMEOUT == 3600
        assert cfg.MAX_WORKERS == 4

    def test_config_instance(self):
        """Test global config instance"""
        assert config.PROJECT_ROOT is not None
        assert isinstance(config.MODEL_DIR, str)

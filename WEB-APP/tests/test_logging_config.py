import pytest
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logging_config import get_logger


class TestLoggingConfig:
    """Test logging configuration"""

    def test_get_logger(self):
        """Test getting a named logger"""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test"

    def test_get_logger_different_names(self):
        """Test getting loggers with different names"""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 is not logger2

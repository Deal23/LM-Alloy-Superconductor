"""Tests for the external virtual environment location."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


script = Path(__file__).resolve().parents[1] / "scripts" / "venv_path.py"
spec = importlib.util.spec_from_file_location("venv_path", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VenvPathTests(unittest.TestCase):
    def test_default_is_outside_project(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            path = module.venv_path()
        self.assertNotIn(script.parents[1], path.parents)
        self.assertEqual(path.parent.name, "venvs")

    def test_override_and_relative_rejection(self):
        override = str(Path.home() / "custom lmsc env")
        with mock.patch.dict(os.environ, {"LMSC_VENV_DIR": override}):
            self.assertEqual(module.venv_path(), Path(override))
        with mock.patch.dict(os.environ, {"LMSC_VENV_DIR": "relative"}):
            with self.assertRaises(ValueError):
                module.venv_path()


if __name__ == "__main__":
    unittest.main()

"""Tests for the Windows per-user command registration."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


@unittest.skipUnless(os.name == "nt", "Windows registration only")
class RegisterLmscTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).resolve().parents[1] / "scripts" / "register_lmsc.py"
        spec = importlib.util.spec_from_file_location("register_lmsc", script)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_registration_and_repeat(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, {"LOCALAPPDATA": directory}), \
                mock.patch.object(self.module.shutil, "which", return_value=None), \
                mock.patch.object(self.module.winreg, "OpenKey") as open_key, \
                mock.patch.object(self.module.winreg, "QueryValueEx", return_value=("existing", 2)), \
                mock.patch.object(self.module.winreg, "SetValueEx") as set_value, \
                mock.patch.object(self.module.ctypes.windll.user32, "SendMessageTimeoutW"):
            open_key.return_value.__enter__.return_value = object()
            self.assertEqual(self.module.main(), 0)
            launcher = Path(directory) / "LMSC" / "bin" / "lmsc.cmd"
            self.assertIn('"%s"' % (Path(__file__).resolve().parents[1] / "run.py"),
                          launcher.read_text(encoding="mbcs"))
            self.assertIn(str(launcher.parent), set_value.call_args.args[-1])
            self.assertEqual(self.module.main(), 0)

    def test_existing_command_is_not_overwritten(self):
        with mock.patch.object(self.module.shutil, "which", return_value=r"C:\other\lmsc.cmd"):
            self.assertEqual(self.module.main(), 1)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quantbullet.reporting.slides.quarto import find_quarto


class TestFindQuarto(unittest.TestCase):
    def test_explicit_file_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "quarto.exe"
            exe.write_text("")
            self.assertEqual(find_quarto(exe), exe)

    def test_folder_with_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bin").mkdir()
            exe = Path(tmp) / "bin" / "quarto.exe"
            exe.write_text("")
            self.assertEqual(find_quarto(tmp), exe)

    def test_environment_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "quarto.exe"
            exe.write_text("")
            with mock.patch.dict(os.environ, {"QUARTO_PATH": str(exe)}):
                self.assertEqual(find_quarto(), exe)

    def test_bad_explicit_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            find_quarto("no/such/quarto")

    def test_nothing_found_explains_install(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("shutil.which", return_value=None), \
                mock.patch("quantbullet.reporting.slides.quarto._quarto_cli_binary", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "QUARTO_PATH"):
                find_quarto()


if __name__ == "__main__":
    unittest.main()

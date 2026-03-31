import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from quantbullet.r.r_session import (
    RSession,
    RSessionManager,
    _resolve_lib_paths,
    _validate_lib_path,
    _validate_r_home,
)


class TestValidateRHome(unittest.TestCase):
    def test_accepts_r_installation_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r_home = Path(tmpdir) / "R-4.4.0"
            (r_home / "bin").mkdir(parents=True)

            self.assertEqual(_validate_r_home(str(r_home)), str(r_home.resolve()))

    def test_rejects_executable_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r_home = Path(tmpdir) / "R-4.4.0"
            bin_dir = r_home / "bin"
            bin_dir.mkdir(parents=True)
            rscript = bin_dir / "Rscript.exe"
            rscript.write_text("", encoding="ascii")

            with self.assertRaisesRegex(ValueError, "installation root"):
                _validate_r_home(str(rscript))

    def test_rejects_directory_without_bin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_bin = Path(tmpdir) / "not-r"
            missing_bin.mkdir()

            with self.assertRaisesRegex(ValueError, "contain a bin/ subdirectory"):
                _validate_r_home(str(missing_bin))


class TestValidateLibPaths(unittest.TestCase):
    def test_accepts_library_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_path = Path(tmpdir) / "win-library" / "4.4"
            lib_path.mkdir(parents=True)

            self.assertEqual(_validate_lib_path(str(lib_path)), str(lib_path.resolve()))

    def test_rejects_missing_library_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "missing"

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                _validate_lib_path(str(missing_path))

    def test_resolve_lib_paths_prepends_r_home_library_and_dedupes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r_home = Path(tmpdir) / "R-4.4.0"
            explicit_lib = Path(tmpdir) / "win-library" / "4.4"
            r_home_library = r_home / "library"
            r_home_library.mkdir(parents=True)
            explicit_lib.mkdir(parents=True)

            resolved = _resolve_lib_paths(
                r_home=str(r_home),
                lib_paths=[str(explicit_lib), str(r_home_library)],
                include_r_home_library=True,
            )

            self.assertEqual(
                resolved,
                [str(r_home_library.resolve()), str(explicit_lib.resolve())],
            )

    def test_resolve_lib_paths_can_exclude_r_home_library(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r_home = Path(tmpdir) / "R-4.4.0"
            explicit_lib = Path(tmpdir) / "win-library" / "4.4"
            explicit_lib.mkdir(parents=True)

            resolved = _resolve_lib_paths(
                r_home=str(r_home),
                lib_paths=[str(explicit_lib)],
                include_r_home_library=False,
            )

            self.assertEqual(resolved, [str(explicit_lib.resolve())])


class TestSourceRFileCaching(unittest.TestCase):
    def test_source_file_only_runs_once_by_default(self):
        source_mock = Mock()

        manager = RSessionManager()
        manager._session = RSession(
            ro=type("FakeRO", (), {"r": {"source": source_mock}})(),
            pandas2ri=object(),
            numpy2ri=object(),
            localconverter=object(),
        )

        r_file = Path("C:/tmp/mgcv.R")
        manager.source_file(r_file)
        manager.source_file(r_file)

        self.assertEqual(source_mock.call_count, 1)
        self.assertEqual(source_mock.call_args.args[0], r_file.resolve().as_posix())

    def test_source_file_can_be_forced(self):
        source_mock = Mock()

        manager = RSessionManager()
        manager._session = RSession(
            ro=type("FakeRO", (), {"r": {"source": source_mock}})(),
            pandas2ri=object(),
            numpy2ri=object(),
            localconverter=object(),
        )

        r_file = Path("C:/tmp/plots.R")
        manager.source_file(r_file)
        manager.source_file(r_file, force=True)

        self.assertEqual(source_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()

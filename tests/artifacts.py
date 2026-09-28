"""Output directories for tests that create inspectable files.

Set QB_TEST_KEEP_ARTIFACTS=1 in the process environment or repository .env to
retain files under tests/_cache_dir. A process environment value takes priority.
Ordinary tests use isolated temporary directories under the ignored cache
root, cleaned by unittest. This also works in restricted test environments.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import unittest
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "tests" / "_cache_dir"
_SETTING = "QB_TEST_KEEP_ARTIFACTS"


class TestTemporaryDirectory:
    """A writable, scoped directory under the ignored test cache root."""

    def __init__(self, prefix: str):
        root = CACHE_ROOT / "_tmp"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{prefix}{uuid4().hex}"
        self.path.mkdir()
        self.name = str(self.path)

    def cleanup(self) -> None:
        # Only remove this instance's direct child of the dedicated temp root.
        root = (CACHE_ROOT / "_tmp").resolve()
        path = self.path.resolve()
        if path.parent != root:
            raise ValueError("temporary test directory escaped its root")
        if path.exists():
            shutil.rmtree(path)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, *args) -> None:
        self.cleanup()


def temporary_artifact_dir(*, prefix: str) -> TestTemporaryDirectory:
    return TestTemporaryDirectory(prefix)


def keep_test_artifacts() -> bool:
    value = os.environ.get(_SETTING)
    if value is None and (REPO_ROOT / ".env").is_file():
        from dotenv import dotenv_values

        value = dotenv_values(REPO_ROOT / ".env").get(_SETTING)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{_SETTING} must be 1 or 0 (also accepts true/false)")


def artifact_dir(test: unittest.TestCase, area: str) -> Path:
    """Return a unique output directory for one unittest method.

    Kept files are grouped by area/class/method; no test deletes or overwrites
    another test's output. Temporary directories are removed by addCleanup,
    including when the test raises before tearDown.
    """
    if keep_test_artifacts():
        path = CACHE_ROOT / area / type(test).__name__ / test._testMethodName
    else:
        temporary = temporary_artifact_dir(prefix="test-")
        test.addCleanup(temporary.cleanup)
        path = Path(temporary.name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def gallery_dir(test_class: type[unittest.TestCase]) -> Path:
    """One stable gallery when retained, isolated temp output otherwise."""
    if keep_test_artifacts():
        path = CACHE_ROOT / "grouped_means"
        path.mkdir(parents=True, exist_ok=True)
        return path
    temporary = temporary_artifact_dir(prefix="gallery-")
    test_class.addClassCleanup(temporary.cleanup)
    return Path(temporary.name)

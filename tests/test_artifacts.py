"""The test artifact policy is independent of the plotting code."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from tests import artifacts


class TestArtifactPolicy(unittest.TestCase):
    def test_env_file_and_process_override(self):
        with artifacts.temporary_artifact_dir(prefix="config-") as temp:
            Path(temp, ".env").write_text("QB_TEST_KEEP_ARTIFACTS=1\n", encoding="utf-8")
            with patch.object(artifacts, "REPO_ROOT", Path(temp)):
                with patch.dict(os.environ, {"QB_TEST_KEEP_ARTIFACTS": "0"}):
                    self.assertFalse(artifacts.keep_test_artifacts())
                with patch.dict(os.environ, {}, clear=True):
                    self.assertTrue(artifacts.keep_test_artifacts())

    def test_temporary_output_is_cleaned_and_kept_output_is_scoped(self):
        case = unittest.TestCase("runTest")
        with patch.object(artifacts, "keep_test_artifacts", return_value=False):
            output = artifacts.artifact_dir(case, "plot/example")
            self.assertTrue(output.is_dir())
            case.doCleanups()
            self.assertFalse(output.exists())

        with artifacts.temporary_artifact_dir(prefix="policy-") as temp:
            with (
                patch.object(artifacts, "keep_test_artifacts", return_value=True),
                patch.object(artifacts, "CACHE_ROOT", Path(temp)),
            ):
                output = artifacts.artifact_dir(case, "plot/example")
                self.assertEqual(output, Path(temp) / "plot/example/TestCase/runTest")
                self.assertTrue(output.is_dir())

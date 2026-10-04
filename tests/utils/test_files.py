"""File helpers stay independent of project workflows."""
import hashlib

import pytest

from quantbullet.utils.files import expand_env_path, file_sha256, temporary_output


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"turnover")
    assert file_sha256(path) == hashlib.sha256(b"turnover").hexdigest()


def test_temporary_output_replaces_on_success_and_keeps_original_on_failure(tmp_path):
    target = tmp_path / "out" / "frame.parquet"
    target.parent.mkdir()
    target.write_text("old", encoding="utf-8")

    with temporary_output(target) as path:
        path.write_text("new", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "new"
    assert list(target.parent.glob(".frame-*")) == []

    with pytest.raises(RuntimeError, match="failed write"):
        with temporary_output(target) as path:
            path.write_text("partial", encoding="utf-8")
            raise RuntimeError("failed write")
    assert target.read_text(encoding="utf-8") == "new"
    assert list(target.parent.glob(".frame-*")) == []


def test_expand_env_path_resolves_variables_and_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("FILES_TEST_ROOT", str(tmp_path))
    assert expand_env_path("${FILES_TEST_ROOT}/panel.parquet", base=tmp_path) == tmp_path / "panel.parquet"
    assert expand_env_path("model", base=tmp_path) == (tmp_path / "model").resolve()

    monkeypatch.delenv("FILES_TEST_ROOT")
    with pytest.raises(ValueError, match="environment variable"):
        expand_env_path("${FILES_TEST_ROOT}/panel.parquet", base=tmp_path)

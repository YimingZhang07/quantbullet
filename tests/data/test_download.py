from dataclasses import replace
from io import BytesIO
from pathlib import Path
import http.client
import json
import urllib.error

import pytest

from quantbullet.data import download as module
from quantbullet.data.download import DownloadSpec, download_source


CSV = b"date,value\n2020-01-01,100\n"


class Response(BytesIO):
    def __init__(self, body: bytes = CSV, **headers):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body)), **headers}


@pytest.fixture
def spec():
    return DownloadSpec("sample", "example", "https://example.invalid/data.csv", "data.csv", {})


@pytest.fixture(autouse=True)
def no_live_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Tests must not access a live data source")

    monkeypatch.setattr(module.urllib.request, "urlopen", reject)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)


def manifest(root: Path) -> dict:
    return json.loads((root / "manifests" / "downloads.json").read_text())


def test_download_cache_refresh_and_history(tmp_path, monkeypatch, spec):
    calls = []
    body = CSV

    def open_response(request, *, timeout):
        calls.append((request.full_url, timeout))
        return Response(body, ETag="example-etag", **{"Last-Modified": "example-date"})

    monkeypatch.setattr(module.urllib.request, "urlopen", open_response)
    progress = []
    first = download_source(spec, tmp_path, progress=lambda size, total: progress.append((size, total)))
    assert first.status == "downloaded"
    assert first.path.read_bytes() == CSV
    assert first.bytes == len(CSV)
    assert progress == [(len(CSV), len(CSV))]
    assert calls == [(spec.url, 30)]
    entry = manifest(tmp_path)["datasets"]["sample"]
    assert entry["current"]["etag"] == "example-etag"
    assert not Path(entry["current"]["path"]).is_absolute()
    assert str(tmp_path) not in (tmp_path / "manifests" / "downloads.json").read_text()
    assert download_source(spec, tmp_path).status == "cached"
    assert len(calls) == 1
    assert download_source(spec, tmp_path, refresh=True).status == "unchanged"
    assert len(manifest(tmp_path)["datasets"]["sample"]["versions"]) == 1

    body = CSV.replace(b"100", b"101")
    second = download_source(spec, tmp_path, refresh=True)
    assert second.status == "downloaded" and second.path != first.path
    assert first.path.read_bytes() == CSV
    assert second.path.read_bytes() == body
    assert len(manifest(tmp_path)["datasets"]["sample"]["versions"]) == 2
    assert not list((tmp_path / "scratch").iterdir())


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_local_file_is_repaired(tmp_path, monkeypatch, spec, damage):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response())
    result = download_source(spec, tmp_path)
    if damage == "missing":
        result.path.unlink()
    else:
        result.path.write_bytes(b"bad")
    repaired = download_source(spec, tmp_path)
    assert repaired.path.read_bytes() == CSV
    assert len(manifest(tmp_path)["datasets"]["sample"]["versions"]) == 1


@pytest.mark.parametrize("body,headers", [
    (b"", {}),
    (b"<html>error</html>", {}),
    (CSV, {"Content-Type": "text/html"}),
    (b"date,value\n", {}),
    (b"date,value\n2020-01-01\n", {}),
])
def test_invalid_response_preserves_previous(tmp_path, monkeypatch, spec, body, headers):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response())
    first = download_source(spec, tmp_path)
    manifest_path = tmp_path / "manifests" / "downloads.json"
    before = manifest_path.read_bytes()
    calls = []

    def bad_response(*a, **k):
        calls.append(1)
        return Response(body, **headers)

    monkeypatch.setattr(module.urllib.request, "urlopen", bad_response)
    with pytest.raises(ValueError):
        download_source(spec, tmp_path, refresh=True)
    assert len(calls) == 1
    assert manifest_path.read_bytes() == before
    assert first.path.read_bytes() == CSV
    assert not list((tmp_path / "scratch").iterdir())


@pytest.mark.parametrize("code,expected_calls", [(404, 1), (429, 3), (503, 3)])
def test_http_retry_policy(tmp_path, monkeypatch, spec, code, expected_calls):
    calls = []
    waits = []

    def error(*a, **k):
        calls.append(1)
        raise urllib.error.HTTPError(spec.url, code, "example error", {}, BytesIO())

    monkeypatch.setattr(module.urllib.request, "urlopen", error)
    monkeypatch.setattr(module.time, "sleep", waits.append)
    with pytest.raises((ValueError, module._RetryableDownloadError)):
        download_source(spec, tmp_path)
    assert len(calls) == expected_calls
    assert waits == ([1, 3] if expected_calls == 3 else [])
    assert not (tmp_path / "manifests" / "downloads.json").exists()
    assert not list((tmp_path / "scratch").iterdir())


def test_interrupted_download_restarts_and_succeeds(tmp_path, monkeypatch, spec):
    class Interrupted(Response):
        def read(self, size):
            if self.tell():
                raise http.client.IncompleteRead(b"", 1)
            return super().read(5)

    responses = iter([Interrupted(), Response()])
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: next(responses))
    result = download_source(spec, tmp_path)
    assert result.path.read_bytes() == CSV


def test_short_download_preserves_manifest(tmp_path, monkeypatch, spec):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response())
    first = download_source(spec, tmp_path)
    before = manifest(tmp_path)
    monkeypatch.setattr(
        module.urllib.request, "urlopen",
        lambda *a, **k: Response(CSV[:10], **{"Content-Length": str(len(CSV))}),
    )
    with pytest.raises(module._RetryableDownloadError):
        download_source(spec, tmp_path, refresh=True)
    assert manifest(tmp_path) == before
    assert first.path.read_bytes() == CSV


def test_connection_retry(tmp_path, monkeypatch, spec):
    calls = []

    def connect(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("example connection failure")
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", connect)
    assert download_source(spec, tmp_path).path.read_bytes() == CSV
    assert len(calls) == 3


def test_manifest_failure_keeps_previous_version(tmp_path, monkeypatch, spec):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response())
    first = download_source(spec, tmp_path)
    before = manifest(tmp_path)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response(CSV.replace(b"100", b"102")))
    original_replace = module.os.replace

    def fail_manifest(source, target):
        if Path(target).name == "downloads.json":
            raise OSError("example manifest write failure")
        return original_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_manifest)
    with pytest.raises(OSError):
        download_source(spec, tmp_path, refresh=True)
    assert manifest(tmp_path) == before
    assert first.path.read_bytes() == CSV
    assert not list((tmp_path / "scratch").iterdir())
    assert not list((tmp_path / "manifests").glob("*.tmp"))


def test_source_change_invalidates_cache(tmp_path, monkeypatch, spec):
    calls = []

    def response(*a, **k):
        calls.append(1)
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", response)
    download_source(spec, tmp_path)
    changed = replace(spec, url="https://example.invalid/new.csv")
    download_source(changed, tmp_path)
    assert len(calls) == 2
    assert manifest(tmp_path)["datasets"]["sample"]["url"] == changed.url
    moved = download_source(replace(changed, provider="other"), tmp_path)
    current = manifest(tmp_path)["datasets"]["sample"]["current"]
    assert tmp_path / current["path"] == moved.path
    assert current["provider"] == "other"


def test_manifest_cannot_select_file_outside_root(tmp_path, monkeypatch, spec):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response())
    download_source(spec, tmp_path)
    state = manifest(tmp_path)
    state["datasets"]["sample"]["current"]["path"] = "../outside.csv"
    (tmp_path / "manifests" / "downloads.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="inside the data root"):
        download_source(spec, tmp_path)


def test_cli_continues_after_failure(monkeypatch, capsys):
    from procs.housing_macro import download as process

    calls = []

    def fetch(source, root, **kwargs):
        calls.append(source.dataset_id)
        if source.dataset_id == "zhvi_metro":
            raise ValueError("example failure")
        return module.DownloadResult(source.dataset_id, "cached", root / "example.csv", "abc", 10)

    monkeypatch.setattr(process, "download_source", fetch)
    # Fetch is mocked: this path is never created or written to.
    external_root = Path(process.__file__).resolve().parents[2].parent / "example-macro-data"
    assert process.main(["--data-root", str(external_root), "--dataset", "zhvi_metro", "cpi"]) == 1
    assert calls == ["zhvi_metro", "cpi"]
    assert "Failed datasets: zhvi_metro" in capsys.readouterr().err


def test_cli_rejects_repository_root(monkeypatch):
    from procs.housing_macro import download as process

    root = Path(process.__file__).resolve().parents[2]
    monkeypatch.setattr(process, "download_source", lambda *a, **k: pytest.fail("Must not download"))
    with pytest.raises(SystemExit) as error:
        process.main(["--data-root", str(root / "local-data")])
    assert error.value.code == 2

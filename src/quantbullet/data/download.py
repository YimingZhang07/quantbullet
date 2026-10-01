"""Stream public CSV downloads into content-addressed local snapshots.

The caller owns the data directory. Downloads are sequential; concurrent
writers to the same manifest are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping
import csv
import hashlib
import http.client
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


HeaderValidator = Callable[[tuple[str, ...]], dict]
ProgressCallback = Callable[[int, int | None], None]
MANIFEST_VERSION = 1
CHUNK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class DownloadSpec:
    dataset_id: str
    provider: str
    url: str
    filename: str
    metadata: Mapping[str, str]
    header_validator: HeaderValidator | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for value in (self.dataset_id, self.provider):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
                raise ValueError("Provider and dataset identifiers must be simple path components")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.csv", self.filename):
            raise ValueError("Filename must be a plain CSV filename")
        parsed = urllib.parse.urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Download URL must use HTTP or HTTPS")
        if parsed.username or parsed.password:
            raise ValueError("Download URL must not include credentials")


@dataclass(frozen=True)
class DownloadResult:
    dataset_id: str
    status: str
    path: Path
    sha256: str
    bytes: int


class _RetryableDownloadError(Exception):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"manifest_version": MANIFEST_VERSION, "datasets": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("Unsupported download manifest version")
    if not isinstance(manifest.get("datasets"), dict):
        raise ValueError("Invalid download manifest datasets")
    return manifest


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".downloads-", suffix=".tmp", delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(manifest, file, indent=2, sort_keys=True, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_csv(path: Path, spec: DownloadSpec) -> dict:
    if not path.stat().st_size:
        raise ValueError("Empty CSV response")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file, strict=True)
            header = tuple(next(reader))
            if not header or any("<" in name or ">" in name for name in header):
                raise ValueError("Response is not a CSV header")
            if len(set(header)) != len(header) or any(not name for name in header):
                raise ValueError("CSV header has empty or duplicate columns")
            first_row = next((row for row in reader if row), None)
            if first_row is None or len(first_row) != len(header):
                raise ValueError("CSV needs a data row matching its header")
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise ValueError("Response is not a readable CSV") from exc
    details = spec.header_validator(header) if spec.header_validator else {}
    return {"status": "passed", "column_count": len(header), **details}


def _download_once(
    spec: DownloadSpec, temporary: Path, progress: ProgressCallback | None,
) -> tuple[str, int, dict]:
    request = urllib.request.Request(spec.url, headers={"User-Agent": "quantbullet-data/0.1"})
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        if code == 429 or 500 <= code < 600:
            raise _RetryableDownloadError(f"HTTP {code}") from exc
        raise ValueError(f"HTTP {code} for {spec.dataset_id}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise _RetryableDownloadError("Could not open download connection") from exc

    with response:
        if "html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("Server returned HTML instead of CSV")
        length = response.headers.get("Content-Length")
        expected = int(length) if length is not None else None
        if expected is not None and expected < 0:
            raise ValueError("Invalid Content-Length")
        digest = hashlib.sha256()
        size = 0
        with temporary.open("wb") as file:
            while True:
                try:
                    chunk = response.read(CHUNK_SIZE)
                except (OSError, http.client.HTTPException) as exc:
                    raise _RetryableDownloadError("Download connection interrupted") from exc
                if not chunk:
                    break
                file.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if progress:
                    progress(size, expected)
            if expected is not None and size != expected:
                raise _RetryableDownloadError("Download size does not match Content-Length")
            file.flush()
            os.fsync(file.fileno())
        headers = {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }
    return digest.hexdigest(), size, headers


def _snapshot_path(root: Path, snapshot: dict) -> Path:
    relative = snapshot.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Manifest snapshot path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest snapshot path must stay inside the data root")
    return path


def download_source(
    spec: DownloadSpec,
    data_root: str | Path,
    *,
    refresh: bool = False,
    progress: ProgressCallback | None = None,
) -> DownloadResult:
    """Download one CSV, preserving past snapshots and publishing on success.

    Default runs reuse a verified local file. ``refresh=True`` fetches the
    complete remote file, even when it exists locally. Header validation is a
    file-format check, not a completeness or economic-data quality assessment.
    """
    root = Path(data_root).expanduser().resolve()
    manifest_path = root / "manifests" / "downloads.json"
    manifest = _load_manifest(manifest_path)
    previous = manifest["datasets"].get(spec.dataset_id)
    identity = {
        "provider": spec.provider, "url": spec.url,
        "filename": spec.filename, "metadata": dict(spec.metadata),
    }
    now = datetime.now(timezone.utc).isoformat()
    current = previous.get("current", {}) if previous else {}
    if previous and not refresh and all(previous.get(k) == v for k, v in identity.items()):
        path = _snapshot_path(root, current)
        if (
            path.is_file() and path.stat().st_size == current.get("bytes")
            and _sha256(path) == current.get("sha256")
        ):
            _validate_csv(path, spec)
            previous["last_checked_at_utc"] = now
            _write_manifest(manifest_path, manifest)
            return DownloadResult(spec.dataset_id, "cached", path, current["sha256"], current["bytes"])

    scratch = root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=scratch, suffix=".part", delete=False) as file:
        temporary = Path(file.name)
    try:
        for attempt in range(3):
            try:
                digest, size, headers = _download_once(spec, temporary, progress)
                break
            except _RetryableDownloadError:
                if attempt == 2:
                    raise
                time.sleep((1, 3)[attempt])
        validation = _validate_csv(temporary, spec)
        relative = Path("raw") / spec.provider / spec.dataset_id / f"{digest}.csv"
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Snapshot destination must stay inside the data root")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or _sha256(target) != digest:
            os.replace(temporary, target)
        downloaded_at = datetime.now(timezone.utc).isoformat()
        versions = list(previous.get("versions", [])) if previous else []
        snapshot = next(
            (item for item in versions if item["sha256"] == digest and item["path"] == relative.as_posix()),
            None,
        )
        if snapshot is None:
            snapshot = {
                "path": relative.as_posix(), "sha256": digest, "bytes": size,
                "downloaded_at_utc": downloaded_at, "validation": validation,
                **identity, **headers,
            }
            versions.append(snapshot)
        manifest["datasets"][spec.dataset_id] = {
            **identity, "current": snapshot, "versions": versions,
            "last_checked_at_utc": downloaded_at,
        }
        _write_manifest(manifest_path, manifest)
        status = "unchanged" if current.get("sha256") == digest else "downloaded"
        return DownloadResult(spec.dataset_id, status, target, digest, size)
    finally:
        temporary.unlink(missing_ok=True)

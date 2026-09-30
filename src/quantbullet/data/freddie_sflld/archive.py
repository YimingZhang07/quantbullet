"""Read Freddie Mac's full -> year -> quarter nested ZIP archive."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import io
import re
import shutil
import struct
import zipfile


YEAR_RE = re.compile(r"historical_data_(\d{4})\.zip\Z")
QUARTER_RE = re.compile(r"historical_data_(\d{4}Q[1-4])\.zip\Z")
VINTAGE_RE = re.compile(r"\d{4}Q[1-4]\Z")


class StoredMemberView(io.RawIOBase):
    """Seekable bounded view over a ZIP_STORED member, without copying it."""

    def __init__(self, parent: io.BufferedIOBase, info: zipfile.ZipInfo):
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"Nested ZIP member must be stored: {info.filename}")
        if info.flag_bits & 0x1:
            raise ValueError(f"Encrypted ZIP member is unsupported: {info.filename}")
        parent.seek(info.header_offset)
        header = parent.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid local ZIP header: {info.filename}")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        self._parent = parent
        self._start = info.header_offset + 30 + name_length + extra_length
        self._size = info.file_size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if position < 0:
            raise ValueError("Cannot seek before beginning of ZIP member")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        remaining = max(self._size - self._position, 0)
        count = remaining if size is None or size < 0 else min(size, remaining)
        self._parent.seek(self._start + self._position)
        data = self._parent.read(count)
        self._position += len(data)
        return data


def sha256_stream(stream: io.RawIOBase, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


class SFLLDArchive:
    """Inspect and extract quarter archives without expanding other vintages."""

    def __init__(self, source: str | Path):
        self.source = Path(source)

    def list_vintages(self) -> list[str]:
        vintages: list[str] = []
        with self.source.open("rb") as source_file, zipfile.ZipFile(source_file) as full:
            for year_info in full.infolist():
                match = YEAR_RE.fullmatch(year_info.filename)
                if not match:
                    raise ValueError(f"Unexpected full archive member: {year_info.filename}")
                with zipfile.ZipFile(StoredMemberView(source_file, year_info)) as year_zip:
                    for quarter_info in year_zip.infolist():
                        quarter_match = QUARTER_RE.fullmatch(quarter_info.filename)
                        if not quarter_match or not quarter_match[1].startswith(match[1]):
                            raise ValueError(f"Unexpected year archive member: {quarter_info.filename}")
                        vintages.append(quarter_match[1])
        if len(set(vintages)) != len(vintages):
            raise ValueError("Duplicate vintage in full archive")
        return sorted(vintages)

    @contextmanager
    def open_quarter(self, vintage: str):
        """Yield (seekable quarter bytes, quarter ZIP) for one vintage."""
        if not VINTAGE_RE.fullmatch(vintage):
            raise ValueError(f"Invalid vintage: {vintage}")
        year_name = f"historical_data_{vintage[:4]}.zip"
        quarter_name = f"historical_data_{vintage}.zip"
        with self.source.open("rb") as source_file, zipfile.ZipFile(source_file) as full:
            try:
                year_info = full.getinfo(year_name)
            except KeyError as exc:
                raise ValueError(f"Year ZIP not found: {year_name}") from exc
            year_view = StoredMemberView(source_file, year_info)
            with zipfile.ZipFile(year_view) as year_zip:
                try:
                    quarter_info = year_zip.getinfo(quarter_name)
                except KeyError as exc:
                    raise ValueError(f"Quarter ZIP not found: {quarter_name}") from exc
                quarter_view = StoredMemberView(year_view, quarter_info)
                with zipfile.ZipFile(quarter_view) as quarter_zip:
                    expected = {f"orig_{vintage}.txt", f"perf_{vintage}.txt"}
                    actual = {info.filename for info in quarter_zip.infolist()}
                    if actual != expected:
                        raise ValueError(
                            f"Unexpected files in {quarter_name}: expected {sorted(expected)}, "
                            f"got {sorted(actual)}"
                        )
                    yield quarter_view, quarter_zip

    def extract_quarter(self, vintage: str, destination: str | Path) -> tuple[str, dict[str, Path]]:
        """Copy one quarter's text files to destination; ZIP CRC is checked on read."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with self.open_quarter(vintage) as (quarter_view, quarter_zip):
            source_hash = sha256_stream(quarter_view)
            paths: dict[str, Path] = {}
            for kind in ("orig", "perf"):
                name = f"{kind}_{vintage}.txt"
                target = destination / name
                with quarter_zip.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                paths[kind] = target
        return source_hash, paths

    def quarter_hash(self, vintage: str) -> str:
        with self.open_quarter(vintage) as (quarter_view, _):
            return sha256_stream(quarter_view)

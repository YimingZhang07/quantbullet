from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import polars as pl
import pytest

from quantbullet.data.freddie_sflld import SFLLDArchive, convert_vintage, load_manifest
from quantbullet.data.freddie_sflld.schema import ORIG_COLUMNS, PERF_COLUMNS


def _zip_bytes(files: dict[str, bytes], *, compression: int = zipfile.ZIP_STORED) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue()


def _row(columns: tuple[str, ...], **values: str) -> bytes:
    return ("|".join(values.get(name, "") for name in columns) + "\n").encode()


def _build_archive(
    path: Path,
    *,
    include_perf: bool = True,
    corrupt_orig: bool = False,
    corrupt_quarter_zip: bool = False,
    extra_loan: bool = False,
) -> None:
    vintage = "2015Q1"
    orig = _row(ORIG_COLUMNS, loan_identifier="F15Q10000001", classic_fico="750")
    if extra_loan:
        orig += _row(ORIG_COLUMNS, loan_identifier="F15Q10000002", classic_fico="710")
    if corrupt_orig:
        orig = b"bad|row\n"
    perf = _row(PERF_COLUMNS, loan_identifier="F15Q10000001", period="201503")
    perf += _row(PERF_COLUMNS, loan_identifier="F15Q10000001", period="201504")
    quarter_members = {f"orig_{vintage}.txt": orig}
    if include_perf:
        quarter_members[f"perf_{vintage}.txt"] = perf
    quarter = _zip_bytes(quarter_members, compression=zipfile.ZIP_DEFLATED)
    if corrupt_quarter_zip:
        quarter = quarter[:-20]
    year = _zip_bytes({f"historical_data_{vintage}.zip": quarter})
    path.write_bytes(_zip_bytes({"historical_data_2015.zip": year}))


def test_convert_and_skip_and_update(tmp_path: Path) -> None:
    source = tmp_path / "full.zip"
    root = tmp_path / "data"
    _build_archive(source)
    archive = SFLLDArchive(source)
    assert archive.list_vintages() == ["2015Q1"]
    first = convert_vintage(archive, "2015Q1", root)
    assert first["status"] == "converted"
    assert first["orig"]["rows"] == 1
    assert first["perf"]["rows"] == 2
    orig = pl.read_parquet(root / first["orig"]["path"])
    perf = pl.read_parquet(root / first["perf"]["path"])
    assert len(orig.columns) == 31 and len(perf.columns) == 35
    assert perf.join(orig, on="loan_identifier").height == 2
    assert orig["classic_fico"].to_list() == ["750"]

    assert convert_vintage(archive, "2015Q1", root)["status"] == "skipped"
    _build_archive(source, extra_loan=True)
    updated = convert_vintage(SFLLDArchive(source), "2015Q1", root)
    assert updated["status"] == "converted"
    assert updated["orig"]["rows"] == 2
    assert not (root / first["orig"]["path"]).exists()
    assert load_manifest(root / "manifests" / "conversion.json")["quarters"]["2015Q1"]["orig"] == updated["orig"]


@pytest.mark.parametrize("missing_perf,corrupt_orig", [(True, False), (False, True)])
def test_failed_update_preserves_previous_result(
    tmp_path: Path, missing_perf: bool, corrupt_orig: bool
) -> None:
    source = tmp_path / "full.zip"
    root = tmp_path / "data"
    _build_archive(source)
    original = convert_vintage(SFLLDArchive(source), "2015Q1", root)
    _build_archive(source, include_perf=not missing_perf, corrupt_orig=corrupt_orig)
    with pytest.raises(ValueError):
        convert_vintage(SFLLDArchive(source), "2015Q1", root)
    current = load_manifest(root / "manifests" / "conversion.json")["quarters"]["2015Q1"]
    assert current["source_sha256"] == original["source_sha256"]
    assert (root / original["orig"]["path"]).is_file()
    assert (root / original["perf"]["path"]).is_file()


def test_bad_outer_zip(tmp_path: Path) -> None:
    source = tmp_path / "broken.zip"
    source.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        SFLLDArchive(source).list_vintages()


def test_bad_nested_zip(tmp_path: Path) -> None:
    source = tmp_path / "broken-quarter.zip"
    _build_archive(source, corrupt_quarter_zip=True)
    with pytest.raises(zipfile.BadZipFile):
        SFLLDArchive(source).quarter_hash("2015Q1")


def test_damaged_parquet_is_rebuilt(tmp_path: Path) -> None:
    source = tmp_path / "full.zip"
    root = tmp_path / "data"
    _build_archive(source)
    archive = SFLLDArchive(source)
    original = convert_vintage(archive, "2015Q1", root)
    (root / original["perf"]["path"]).write_bytes(b"broken")
    updated = convert_vintage(archive, "2015Q1", root)
    assert updated["status"] == "converted"
    assert updated["perf"]["path"] != original["perf"]["path"]
    assert updated["perf"]["rows"] == 2

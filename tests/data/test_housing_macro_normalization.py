from datetime import date
import errno
import hashlib
import json

import polars as pl
import pytest

from procs.housing_macro import build_parquet as module
from quantbullet.data.fred import fred_csv_source, scan_cpi_csv
from quantbullet.data.zillow import scan_zhvi_csv, zhvi_sources


def _sources(root, *, metro=None, state=None, zipcode=None, cpi=None):
    bodies = {
        "zhvi_metro": metro or "RegionID,RegionName,RegionType,SizeRank,2020-01-31,2020-03-31\n1,United States,country,0,100,120\n2,Test Metro,msa,1,,130\n",
        "zhvi_state": state or "RegionID,RegionName,RegionType,2020-01-31,2020-03-31\n3,Test State,state,100,110\n",
        "zhvi_zip": zipcode or "RegionID,RegionName,RegionType,State,City,Metro,CountyName,2020-01-31,2020-03-31\n4,00123,zip,TS,Test City,Test Metro,Test County,50,\n",
        "cpi": cpi or "observation_date,CPIAUCNS\n2020-01-15,200\n2020-02-29,.\n2020-03-01,210\n",
    }
    manifest = {"manifest_version": 1, "datasets": {}}
    for spec in (*zhvi_sources(), fred_csv_source("CPIAUCNS")):
        content = bodies[spec.dataset_id].encode()
        digest = hashlib.sha256(content).hexdigest()
        relative = f"raw/{spec.provider}/{spec.dataset_id}/{digest}.csv"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest["datasets"][spec.dataset_id] = {
            "provider": spec.provider, "metadata": dict(spec.metadata),
            "current": {"path": relative, "sha256": digest, "bytes": len(content)},
        }
    (root / "manifests").mkdir(exist_ok=True)
    (root / "manifests/downloads.json").write_text(json.dumps(manifest))
    return manifest


def test_readers_preserve_nulls_codes_and_months(tmp_path):
    manifest = _sources(tmp_path)
    def path(name):
        return tmp_path / manifest["datasets"][name]["current"]["path"]
    zip_frame = scan_zhvi_csv(path("zhvi_zip"), geography="zip").collect()
    assert zip_frame["region_name"].to_list() == ["00123", "00123"]
    assert zip_frame["region_id"].dtype == pl.String
    assert zip_frame["value"].to_list() == [50.0, None]
    assert zip_frame["month"].to_list() == [date(2020, 1, 1), date(2020, 3, 1)]
    assert zip_frame["state_name"].null_count() == 2
    assert zip_frame["county_name"].to_list() == ["Test County"] * 2
    metro = scan_zhvi_csv(path("zhvi_metro"), geography="metro").collect()
    assert metro.filter(pl.col("region_id") == "1")["geography_level"].to_list() == ["national"] * 2
    cpi = scan_cpi_csv(path("cpi")).collect()
    assert cpi.schema == {"provider": pl.String, "series_id": pl.String, "month": pl.Date, "value": pl.Float64}
    assert cpi["value"].to_list() == [200.0, None, 210.0]
    assert cpi["month"].to_list() == [date(2020, n, 1) for n in (1, 2, 3)]


def test_build_coverage_cache_and_corrupt_output(tmp_path):
    _sources(tmp_path)
    result = module.build_parquet(tmp_path)
    assert result["status"] == "built"
    assert result["outputs"]["hpi"]["rows"] == 8
    assert result["outputs"]["cpi"]["rows"] == 3
    report = json.loads((tmp_path / result["reports"]["coverage.json"]["path"]).read_text())
    national = next(row for row in report["hpi"]["full_history"]["regions"] if row["geography_level"] == "national")
    assert national["missing_date_months"] == 1
    assert national["missing_date_months_between_valid_bounds"] == 1
    assert national["null_months_between_valid_bounds"] == 0
    cpi = report["cpi"]["full_history"]["regions"][0]
    assert cpi["missing_date_months"] == 0
    assert cpi["null_months_between_valid_bounds"] == 1
    assert cpi["missing_values_between_valid_bounds"] == 1
    monthly = report["hpi"]["full_history"]["monthly"]
    assert all(row["missing_date_regions"] == 1 for row in monthly if row["month"] == "2020-02-01")
    assert module.build_parquet(tmp_path)["status"] == "cached"
    (tmp_path / result["outputs"]["hpi"]["path"]).write_bytes(b"corrupt")
    rebuilt = module.build_parquet(tmp_path)
    assert rebuilt["status"] == "built" and rebuilt["build_id"] != result["build_id"]


@pytest.mark.parametrize("body", [
    "observation_date,CPIAUCNS\n2020-01-01,not-a-number\n",
    "observation_date,CPIAUCNS\n2020-01-01,inf\n",
    "observation_date,CPIAUCNS\n2020-01-01,NaN\n",
    "observation_date,CPIAUCNS\ninvalid,100\n",
    "observation_date,CPIAUCNS\n,100\n",
    "observation_date,CPIAUCNS\n2020-01-01,100\n2020-01-31,101\n",
])
def test_invalid_cpi_preserves_previous_build(tmp_path, body):
    _sources(tmp_path)
    old = module.build_parquet(tmp_path)
    manifest_path = tmp_path / "manifests/normalization.json"
    before = manifest_path.read_bytes()
    _sources(tmp_path, cpi=body)
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        module.build_parquet(tmp_path)
    assert manifest_path.read_bytes() == before
    assert pl.read_parquet(tmp_path / old["outputs"]["cpi"]["path"]).height == 3
    assert list((tmp_path / "scratch").iterdir()) == []


@pytest.mark.parametrize("body", [
    "RegionID,RegionName,RegionType,2020-01-31\n1,United States,country,100\n1,United States,country,101\n",
    "RegionID,RegionName,RegionType,2020-01-31\n,United States,country,100\n",
    "RegionID,RegionName,RegionType,2020-01-31\n1,United States,unknown,100\n",
    "RegionID,RegionName,RegionType,2020-01-31\n1,United States,country,inf\n",
    "RegionID,RegionName,RegionType,2020-01-31\n1,United States,country,bad\n",
    "RegionID,RegionName,RegionType,2020-01-31\n1,United States,country,.\n",
    "RegionID,RegionName,RegionType,2020-01-01,2020-01-31\n1,United States,country,100,101\n",
    "RegionID,RegionName,RegionType,2020-02-31\n1,United States,country,100\n",
])
def test_invalid_hpi_fails(tmp_path, body):
    _sources(tmp_path, metro=body)
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        module.build_parquet(tmp_path)
    assert not (tmp_path / "manifests/normalization.json").exists()
    assert list((tmp_path / "scratch").iterdir()) == []


def test_staging_cleanup_retries_directory_not_empty(tmp_path, monkeypatch):
    class TemporaryDirectory:
        name = str(tmp_path)
        attempts = 0

        def cleanup(self):
            self.attempts += 1
            if self.attempts < 3:
                raise OSError(errno.ENOTEMPTY, "Writer has not released its directory")

    temporary = TemporaryDirectory()
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda **kwargs: temporary)
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    with pytest.raises(ValueError, match="conversion failed"):
        with module._staging_directory(tmp_path):
            raise ValueError("conversion failed")
    assert temporary.attempts == 3


def test_changed_source_and_report_failure_keep_previous(tmp_path, monkeypatch):
    _sources(tmp_path)
    old = module.build_parquet(tmp_path)
    before = (tmp_path / "manifests/normalization.json").read_bytes()
    _sources(tmp_path, cpi="observation_date,CPIAUCNS\n2020-01-01,222.123456789\n2020-03-01,0\n")
    original = module._markdown
    monkeypatch.setattr(module, "_markdown", lambda report: (_ for _ in ()).throw(RuntimeError("report failure")))
    with pytest.raises(RuntimeError, match="report failure"):
        module.build_parquet(tmp_path)
    assert (tmp_path / "manifests/normalization.json").read_bytes() == before
    monkeypatch.setattr(module, "_markdown", original)
    updated = module.build_parquet(tmp_path)
    assert updated["build_id"] != old["build_id"]
    frame = pl.read_parquet(tmp_path / updated["outputs"]["cpi"]["path"])
    assert frame["value"].to_list() == [222.123456789, 0.0]
    report = json.loads((tmp_path / updated["reports"]["coverage.json"]["path"]).read_text())
    assert report["cpi"]["full_history"]["summary"][0]["nonpositive_values"] == 1
    assert report["cpi"]["full_history"]["summary"][0]["missing_date_months"] == 1
    assert (tmp_path / old["outputs"]["cpi"]["path"]).is_file()


def test_source_hash_checked_even_when_cached(tmp_path):
    sources = _sources(tmp_path)
    module.build_parquet(tmp_path)
    path = tmp_path / sources["datasets"]["cpi"]["current"]["path"]
    path.write_text("modified")
    with pytest.raises(ValueError, match="hash/size"):
        module.build_parquet(tmp_path)


def test_coverage_distinguishes_internal_nulls_from_absent_months(tmp_path):
    _sources(
        tmp_path,
        metro="RegionID,RegionName,RegionType,2014-12-31,2015-02-28,2015-03-31\n1,United States,country,100,,120\n2,Test Metro,msa,,,\n",
        state="RegionID,RegionName,RegionType,2015-01-31,2015-03-31\n3,Test State,state,100,110\n",
        zipcode="RegionID,RegionName,RegionType,2014-12-31,2015-03-31\n4,00123,zip,50,60\n",
    )
    result = module.build_parquet(tmp_path)
    report = json.loads((tmp_path / result["reports"]["coverage.json"]["path"]).read_text())
    rows = {row["geography_level"]: row for row in report["hpi"]["full_history"]["regions"]}
    assert rows["national"]["missing_date_months_between_valid_bounds"] == 1
    assert rows["national"]["null_months_between_valid_bounds"] == 1
    assert rows["national"]["missing_values_between_valid_bounds"] == 2
    assert rows["metro"]["first_valid_month"] is None
    assert rows["metro"]["missing_values_between_valid_bounds"] is None
    assert rows["metro"]["null_values"] == 3
    assert rows["state"]["missing_date_months"] == 2
    recent = {row["geography_level"]: row for row in report["hpi"]["from_2015"]["regions"]}
    assert recent["zip"]["rows"] == 1
    assert recent["zip"]["missing_date_months"] == 2
    assert report["hpi"]["from_2015"]["start_month"] == "2015-01-01"


def test_cli_rejects_repository_data_root():
    with pytest.raises(SystemExit) as error:
        module.main(["--data-root", str(module.Path(__file__).resolve().parents[2] / "local-data")])
    assert error.value.code == 2

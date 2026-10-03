from datetime import date, datetime, timezone
import errno
import hashlib
import json

import polars as pl
import pytest

from procs.housing_macro import build_parquet as module
from procs.housing_macro import coverage
from quantbullet.data.fred import fred_csv_source, scan_cpi_csv
from quantbullet.data.zillow import scan_zhvi_csv, zhvi_sources


def _sources(root, *, metro=None, state=None, zipcode=None, cpi=None, pmms=None):
    bodies = {
        "zhvi_metro": metro or "RegionID,RegionName,RegionType,SizeRank,2020-01-31,2020-03-31\n1,United States,country,0,100,120\n2,Test Metro,msa,1,,130\n",
        "zhvi_state": state or "RegionID,RegionName,RegionType,2020-01-31,2020-03-31\n3,Test State,state,100,110\n",
        "zhvi_zip": zipcode or "RegionID,RegionName,RegionType,State,City,Metro,CountyName,2020-01-31,2020-03-31\n4,00123,zip,TS,Test City,Test Metro,Test County,50,\n",
        "cpi": cpi or "observation_date,CPIAUCNS\n2020-01-15,200\n2020-02-29,.\n2020-03-01,210\n",
        "pmms_30y": pmms or "observation_date,MORTGAGE30US\n2020-01-02,4\n2020-01-31,6\n2020-02-06,.\n2020-02-27,\n2020-03-05,7\n",
    }
    manifest = {"manifest_version": 1, "datasets": {}}
    for spec in (*zhvi_sources(), fred_csv_source("CPIAUCNS"), fred_csv_source("MORTGAGE30US")):
        content = bodies[spec.dataset_id].encode()
        digest = hashlib.sha256(content).hexdigest()
        relative = f"raw/{spec.provider}/{spec.dataset_id}/{digest}.csv"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest["datasets"][spec.dataset_id] = {
            "provider": spec.provider, "metadata": dict(spec.metadata),
            "current": {"path": relative, "sha256": digest, "bytes": len(content),
                        "downloaded_at_utc": "2020-03-15T12:00:00+00:00"},
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


def test_build_fixed_outputs_and_rebuild(tmp_path, monkeypatch):
    _sources(tmp_path)
    legacy_manifest = tmp_path / "manifests/normalization.json"
    legacy_manifest.write_text('{"legacy": true}')
    result = module.build_parquet(tmp_path)
    assert result["hpi"]["rows"] == 8
    assert result["cpi"]["rows"] == 3
    assert result["pmms"]["rows"] == 2
    assert result["hpi"]["first_month"] == date(2020, 1, 1)
    assert sorted(path.name for path in (tmp_path / "parquet").iterdir()) == ["cpi.parquet", "hpi.parquet", "pmms.parquet"]
    assert not (tmp_path / "reports").exists()
    assert legacy_manifest.read_text() == '{"legacy": true}'
    original_sink = pl.LazyFrame.sink_parquet
    calls = []

    def sink(frame, *args, **kwargs):
        calls.append(1)
        return original_sink(frame, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", sink)
    (tmp_path / "parquet/hpi.parquet").write_bytes(b"corrupt")
    module.build_parquet(tmp_path)
    assert len(calls) == 3  # Every run reconstructs all selected outputs.
    assert pl.read_parquet(tmp_path / "parquet/hpi.parquet").height == 8


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
    module.build_parquet(tmp_path)
    before = {name: (tmp_path / f"parquet/{name}.parquet").read_bytes() for name in ("hpi", "cpi")}
    _sources(tmp_path, cpi=body)
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        module.build_parquet(tmp_path)
    for name, content in before.items():
        assert (tmp_path / f"parquet/{name}.parquet").read_bytes() == content
    assert not list((tmp_path / "parquet").glob("normalize-*"))


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
    assert not (tmp_path / "parquet/hpi.parquet").exists()
    assert not (tmp_path / "parquet/cpi.parquet").exists()
    assert not list((tmp_path / "parquet").glob("normalize-*"))


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


def test_conversion_failure_keeps_both_outputs(tmp_path, monkeypatch):
    _sources(tmp_path)
    module.build_parquet(tmp_path)
    before = {name: (tmp_path / f"parquet/{name}.parquet").read_bytes() for name in ("hpi", "cpi")}
    original_sink = pl.LazyFrame.sink_parquet

    def fail_cpi(frame, path, **kwargs):
        if path.name == "cpi.parquet":
            raise OSError("example write failure")
        return original_sink(frame, path, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail_cpi)
    with pytest.raises(OSError, match="example write failure"):
        module.build_parquet(tmp_path)
    for name, content in before.items():
        assert (tmp_path / f"parquet/{name}.parquet").read_bytes() == content
    assert not list((tmp_path / "parquet").glob("normalize-*"))


def test_changed_source_and_separate_coverage(tmp_path):
    _sources(tmp_path)
    module.build_parquet(tmp_path)
    _sources(tmp_path, cpi="observation_date,CPIAUCNS\n2020-01-01,222.123456789\n2020-03-01,0\n")
    module.build_parquet(tmp_path)
    frame = pl.read_parquet(tmp_path / "parquet/cpi.parquet")
    assert frame["value"].to_list() == [222.123456789, 0.0]
    report = coverage.write_coverage(tmp_path)
    summary = report["cpi"]["full_history"]["summary"][0]
    assert summary["nonpositive_values"] == 1
    assert summary["missing_date_months"] == 1
    assert (tmp_path / "reports/coverage.md").is_file()
    assert "schema_version" not in report


@pytest.mark.parametrize("relative", ["../outside.csv", None])
def test_manifest_cannot_select_file_outside_root(tmp_path, relative):
    sources = _sources(tmp_path)
    sources["datasets"]["cpi"]["current"]["path"] = relative or str(tmp_path.parent / "outside.csv")
    (tmp_path / "manifests/downloads.json").write_text(json.dumps(sources))
    with pytest.raises(ValueError, match="inside the data root"):
        module.build_parquet(tmp_path)


def test_coverage_distinguishes_internal_nulls_from_absent_months(tmp_path):
    _sources(
        tmp_path,
        metro="RegionID,RegionName,RegionType,2014-12-31,2015-02-28,2015-03-31\n1,United States,country,100,,120\n2,Test Metro,msa,,,\n",
        state="RegionID,RegionName,RegionType,2015-01-31,2015-03-31\n3,Test State,state,100,110\n",
        zipcode="RegionID,RegionName,RegionType,2014-12-31,2015-03-31\n4,00123,zip,50,60\n",
    )
    module.build_parquet(tmp_path)
    report = coverage.write_coverage(tmp_path)
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
    assert report["hpi"]["from_2015"]["start_month"] == date(2015, 1, 1)
    saved = json.loads((tmp_path / "reports/coverage.json").read_text())
    assert saved["hpi"]["from_2015"]["start_month"] == "2015-01-01"


def test_cli_rejects_repository_data_root():
    with pytest.raises(SystemExit) as error:
        module.main(["--data-root", str(module.Path(__file__).resolve().parents[2] / "local-data")])
    assert error.value.code == 2


def test_pmms_only_build_reads_only_selected_snapshot(tmp_path):
    manifest = _sources(tmp_path)
    module.build_parquet(tmp_path)
    before = {name: (tmp_path / f"parquet/{name}.parquet").read_bytes() for name in ("hpi", "cpi")}
    manifest["datasets"] = {"pmms_30y": manifest["datasets"]["pmms_30y"]}
    (tmp_path / "manifests/downloads.json").write_text(json.dumps(manifest))
    result = module.build_parquet(tmp_path, datasets=["pmms", "pmms"])
    assert list(result) == ["pmms"]
    frame = pl.read_parquet(tmp_path / "parquet/pmms.parquet")
    assert frame["value"].to_list() == [5.0, None]
    for name, content in before.items():
        assert (tmp_path / f"parquet/{name}.parquet").read_bytes() == content
    assert module.main(["--data-root", str(tmp_path), "--dataset", "pmms"]) == 0


def test_pmms_cutoff_uses_snapshot_acquisition_and_build_date(tmp_path, monkeypatch):
    manifest = _sources(tmp_path)
    manifest["datasets"]["pmms_30y"]["last_checked_at_utc"] = "2020-04-15T12:00:00+00:00"
    (tmp_path / "manifests/downloads.json").write_text(json.dumps(manifest))

    class BuildClock(datetime):
        @classmethod
        def now(cls, tz):
            return datetime(2020, 4, 15, tzinfo=timezone.utc)

    monkeypatch.setattr(module, "datetime", BuildClock)
    module.build_parquet(tmp_path, datasets=["pmms"])
    assert pl.read_parquet(tmp_path / "parquet/pmms.parquet")["month"].to_list() == [date(2020, 1, 1), date(2020, 2, 1)]
    monkeypatch.setattr(BuildClock, "now", classmethod(lambda cls, tz: datetime(2020, 2, 15, tzinfo=timezone.utc)))
    module.build_parquet(tmp_path, datasets=["pmms"])
    assert pl.read_parquet(tmp_path / "parquet/pmms.parquet")["month"].to_list() == [date(2020, 1, 1)]


@pytest.mark.parametrize("body", [
    "observation_date,MORTGAGE30US\n2020-01-02,4\n2020-01-02,5\n",
    "observation_date,MORTGAGE30US\n2020-01-02,inf\n",
])
def test_invalid_pmms_preserves_all_outputs(tmp_path, body):
    _sources(tmp_path)
    module.build_parquet(tmp_path)
    before = {name: (tmp_path / f"parquet/{name}.parquet").read_bytes() for name in ("hpi", "cpi", "pmms")}
    _sources(tmp_path, pmms=body)
    with pytest.raises(ValueError):
        module.build_parquet(tmp_path)
    for name, content in before.items():
        assert (tmp_path / f"parquet/{name}.parquet").read_bytes() == content


def test_pmms_write_failure_preserves_all_outputs(tmp_path, monkeypatch):
    _sources(tmp_path)
    module.build_parquet(tmp_path)
    before = {name: (tmp_path / f"parquet/{name}.parquet").read_bytes() for name in ("hpi", "cpi", "pmms")}
    original_sink = pl.LazyFrame.sink_parquet

    def fail_pmms(frame, path, **kwargs):
        if path.name == "pmms.parquet":
            raise OSError("PMMS write failure")
        return original_sink(frame, path, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", fail_pmms)
    with pytest.raises(OSError, match="PMMS write failure"):
        module.build_parquet(tmp_path)
    for name, content in before.items():
        assert (tmp_path / f"parquet/{name}.parquet").read_bytes() == content
    assert not list((tmp_path / "parquet").glob("normalize-*"))


def test_legacy_build_and_optional_pmms_coverage(tmp_path):
    manifest = _sources(tmp_path)
    del manifest["datasets"]["pmms_30y"]
    (tmp_path / "manifests/downloads.json").write_text(json.dumps(manifest))
    module.build_parquet(tmp_path, datasets=["hpi", "cpi"])
    assert set(coverage.write_coverage(tmp_path)) == {"hpi", "cpi"}
    with pytest.raises(ValueError, match="pmms_30y: missing current snapshot"):
        module.build_parquet(tmp_path)
    _sources(tmp_path)
    module.build_parquet(tmp_path, datasets=["pmms"])
    report = coverage.write_coverage(tmp_path)
    assert report["pmms"]["full_history"]["summary"][0]["rows"] == 2
    assert report["pmms"]["from_2015"]["summary"][0]["null_values"] == 1
    assert "PMMS — full_history" in (tmp_path / "reports/coverage.md").read_text(encoding="utf-8")

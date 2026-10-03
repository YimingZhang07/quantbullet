from io import BytesIO
from datetime import date

import polars as pl
import pytest

from quantbullet.data import download as module
from quantbullet.data.download import download_source
from quantbullet.data.fred import aggregate_pmms_monthly, fred_csv_source, scan_pmms_csv


def test_fred_cpi_definition():
    source = fred_csv_source("CPIAUCNS")
    assert source.dataset_id == "cpi"
    assert source.url.endswith("?id=CPIAUCNS")
    assert source.metadata["original_provider"] == "BLS"
    assert source.metadata["seasonal_adjustment"] == "NSA"
    assert fred_csv_source("CPIAUCSL").dataset_id == "fred_cpiaucsl"
    assert source.header_validator(("observation_date", "CPIAUCNS"))["series_id"] == "CPIAUCNS"


@pytest.mark.parametrize("header", [("date", "CPIAUCNS"), ("observation_date", "CPIAUCSL")])
def test_fred_rejects_wrong_header(header):
    with pytest.raises(ValueError):
        fred_csv_source("CPIAUCNS").header_validator(header)


def test_fred_wrong_series_does_not_publish(tmp_path, monkeypatch):
    body = b"observation_date,CPIAUCSL\n2020-01-01,100\n"

    class Response(BytesIO):
        headers = {"Content-Length": str(len(body))}

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response(body))
    with pytest.raises(ValueError):
        download_source(fred_csv_source("CPIAUCNS"), tmp_path)
    assert not (tmp_path / "manifests" / "downloads.json").exists()


def test_fred_invalid_identifier():
    with pytest.raises(ValueError):
        fred_csv_source("CPIAUCNS&other=value")


def test_pmms_source_definition():
    source = fred_csv_source("MORTGAGE30US")
    assert source.dataset_id == "pmms_30y"
    assert source.url.endswith("?id=MORTGAGE30US")
    assert source.filename == "MORTGAGE30US.csv"
    assert source.metadata["original_provider"] == "Freddie Mac"
    assert source.metadata["frequency"] == "weekly"
    assert source.metadata["units"] == "percent"
    assert source.metadata["geography"] == "national"
    assert source.metadata["seasonal_adjustment"] == "NSA"
    assert source.header_validator(("observation_date", "MORTGAGE30US"))["series_id"] == "MORTGAGE30US"


def test_pmms_monthly_mean_dates_nulls_and_cutoff(tmp_path):
    path = tmp_path / "weekly.csv"
    path.write_text("observation_date,MORTGAGE30US\n2020-01-02,4\n2020-01-31,6\n2020-01-15,.\n2020-02-05,\n2020-02-27,.\n2020-03-05,7\n")
    weekly = scan_pmms_csv(path)
    raw = weekly.collect()
    assert raw.schema == {"provider": pl.String, "series_id": pl.String, "observation_date": pl.Date, "value": pl.Float64}
    assert raw["observation_date"].to_list()[1] == date(2020, 1, 31)
    assert raw["value"].to_list() == [4.0, 6.0, None, None, None, 7.0]
    monthly = aggregate_pmms_monthly(weekly, as_of=date(2020, 3, 15)).collect()
    assert monthly.schema == {"provider": pl.String, "series_id": pl.String, "month": pl.Date, "value": pl.Float64}
    assert monthly["month"].to_list() == [date(2020, 1, 1), date(2020, 2, 1)]
    assert monthly["value"].to_list() == [5.0, None]
    assert monthly["provider"].to_list() == ["freddie_mac"] * 2
    assert aggregate_pmms_monthly(weekly, as_of=date(2020, 1, 1)).collect().height == 0


@pytest.mark.parametrize("body", [
    "observation_date,MORTGAGE30US\n2020-01-02,bad\n",
    "observation_date,MORTGAGE30US\n2020-01-02,inf\n",
    "observation_date,MORTGAGE30US\n2020-01-02,NaN\n",
    "observation_date,MORTGAGE30US\n2020-02-31,5\n",
    "observation_date,MORTGAGE30US\n,5\n",
    "observation_date,MORTGAGE30US\n2020-01-02,5\n2020-01-02,6\n",
    "observation_date,MORTGAGE30US\n",
    "observation_date,MORTGAGE15US\n2020-01-02,5\n",
    "observation_date,MORTGAGE30US,MORTGAGE30US\n2020-01-02,5,5\n",
    # Even the excluded current month must be validated before aggregation.
    "observation_date,MORTGAGE30US\n2020-01-02,5\n2020-03-05,inf\n",
    "observation_date,MORTGAGE30US\n2020-03-05,5\n2020-03-05,6\n",
])
def test_pmms_bad_weekly_data_fails(tmp_path, body):
    path = tmp_path / "weekly.csv"
    path.write_text(body)
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        aggregate_pmms_monthly(scan_pmms_csv(path), as_of=date(2020, 3, 15)).collect()


def test_pmms_reader_rejects_other_series(tmp_path):
    with pytest.raises(ValueError, match="MORTGAGE30US only"):
        scan_pmms_csv(tmp_path / "unused.csv", series_id="MORTGAGE15US")

from io import BytesIO

import pytest

from quantbullet.data import download as module
from quantbullet.data.download import download_source
from quantbullet.data.fred import fred_csv_source


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

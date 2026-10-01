from io import BytesIO
import json

import pytest

from quantbullet.data import download as module
from quantbullet.data.download import download_source
from quantbullet.data.zillow import zhvi_sources


def test_zhvi_definitions_and_header():
    sources = zhvi_sources()
    assert [s.dataset_id for s in sources] == ["zhvi_metro", "zhvi_state", "zhvi_zip"]
    assert [s.filename.split("_")[0] for s in sources] == ["Metro", "State", "Zip"]
    for source in sources:
        assert source.url.endswith(source.filename)
        assert "sm_sa_month" in source.url
        summary = source.header_validator(("RegionID", "RegionName", "RegionType", "2020-01-31", "2020-02-29"))
        assert summary == {"date_column_count": 2, "first_month": "2020-01-31", "last_month": "2020-02-29"}


@pytest.mark.parametrize("header", [
    ("RegionID", "RegionName", "2020-01-31"),
    ("RegionID", "RegionName", "RegionType"),
    ("RegionID", "RegionName", "RegionType", "2020-02-31"),
])
def test_zhvi_rejects_wrong_header(header):
    with pytest.raises(ValueError):
        zhvi_sources()[0].header_validator(header)


def test_zip_snapshot_keeps_original_codes(tmp_path, monkeypatch):
    body = b"RegionID,RegionName,RegionType,2020-01-31\n1,00123,zip,100\n"

    class Response(BytesIO):
        headers = {"Content-Length": str(len(body))}

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response(body))
    result = download_source(zhvi_sources()[2], tmp_path)
    assert result.path.read_bytes() == body
    manifest = json.loads((tmp_path / "manifests" / "downloads.json").read_text())
    assert manifest["datasets"]["zhvi_zip"]["current"]["validation"]["first_month"] == "2020-01-31"

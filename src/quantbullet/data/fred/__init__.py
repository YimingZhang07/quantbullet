"""FRED public CSV download definitions, without API credentials."""

from functools import partial
import re
from urllib.parse import urlencode

from quantbullet.data.download import DownloadSpec


def _validate_fred_header(header: tuple[str, ...], *, series_id: str) -> dict:
    if "observation_date" not in header or series_id not in header:
        raise ValueError(f"FRED CSV needs observation_date and {series_id}")
    return {"series_id": series_id, "date_column": "observation_date"}


def fred_csv_source(series_id: str) -> DownloadSpec:
    """Describe a complete-history FRED CSV; CPIAUCNS uses dataset ID cpi."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", series_id):
        raise ValueError("Invalid FRED series identifier")
    metadata = {"series_id": series_id, "distributor": "FRED"}
    if series_id == "CPIAUCNS":
        metadata.update({
            "original_provider": "BLS", "frequency": "monthly", "geography": "national",
            "seasonal_adjustment": "NSA", "units": "index_1982_1984_100",
        })
    return DownloadSpec(
        dataset_id="cpi" if series_id == "CPIAUCNS" else f"fred_{series_id.lower()}",
        provider="fred", filename=f"{series_id}.csv",
        url="https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode({"id": series_id}),
        metadata=metadata,
        header_validator=partial(_validate_fred_header, series_id=series_id),
    )


from .reader import scan_cpi_csv

__all__ = ["fred_csv_source", "scan_cpi_csv"]

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
    """Describe a complete-history FRED CSV, including CPI and weekly PMMS."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", series_id):
        raise ValueError("Invalid FRED series identifier")
    metadata = {"series_id": series_id, "distributor": "FRED"}
    dataset_id = f"fred_{series_id.lower()}"
    if series_id == "CPIAUCNS":
        dataset_id = "cpi"
        metadata.update({
            "original_provider": "BLS", "frequency": "monthly", "geography": "national",
            "seasonal_adjustment": "NSA", "units": "index_1982_1984_100",
        })
    elif series_id == "MORTGAGE30US":
        dataset_id = "pmms_30y"
        metadata.update({
            "original_provider": "Freddie Mac", "frequency": "weekly", "geography": "national",
            "seasonal_adjustment": "NSA", "units": "percent", "mortgage_term_months": "360",
        })
    return DownloadSpec(
        dataset_id=dataset_id,
        provider="fred", filename=f"{series_id}.csv",
        url="https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode({"id": series_id}),
        metadata=metadata,
        header_validator=partial(_validate_fred_header, series_id=series_id),
    )


from .reader import aggregate_pmms_monthly, scan_cpi_csv, scan_pmms_csv

__all__ = ["fred_csv_source", "scan_cpi_csv", "scan_pmms_csv", "aggregate_pmms_monthly"]

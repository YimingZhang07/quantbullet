"""Public Zillow monthly housing-data download definitions."""

from datetime import date
import re

from quantbullet.data.download import DownloadSpec


def _validate_zhvi_header(header: tuple[str, ...]) -> dict:
    required = {"RegionID", "RegionName", "RegionType"}
    if not required.issubset(header):
        raise ValueError("ZHVI CSV is missing required geographic columns")
    months = sorted(name for name in header if re.fullmatch(r"\d{4}-\d{2}-\d{2}", name))
    if not months:
        raise ValueError("ZHVI CSV has no monthly date columns")
    for month in months:
        date.fromisoformat(month)
    return {"date_column_count": len(months), "first_month": months[0], "last_month": months[-1]}


def zhvi_sources() -> tuple[DownloadSpec, ...]:
    """All Homes, mid-tier, smoothed and seasonally adjusted monthly ZHVI.

    The metro CSV includes the national row. ZIP data uses full five-digit
    postal codes; preserving the CSV avoids losing leading zeroes.
    """
    suffix = "zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
    return tuple(
        DownloadSpec(
            dataset_id=f"zhvi_{level}", provider="zillow",
            url=f"https://files.zillowstatic.com/research/public_csvs/zhvi/{prefix}_{suffix}",
            filename=f"{prefix}_{suffix}",
            metadata={
                "metric": "ZHVI", "geography": level, "frequency": "monthly",
                "housing_type": "all_homes_sfr_condo_coop", "tier": "0.33_0.67",
                "smoothing": "smoothed", "seasonal_adjustment": "SA", "units": "USD",
            },
            header_validator=_validate_zhvi_header,
        )
        for level, prefix in (("metro", "Metro"), ("state", "State"), ("zip", "Zip"))
    )


from .reader import scan_zhvi_csv

__all__ = ["zhvi_sources", "scan_zhvi_csv"]

"""Sample loans by vintage and read their complete monthly performance panels."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import random

import polars as pl


def filter_orig_loans(
    orig: pl.LazyFrame, *, amortization_type: str | None = None,
    original_loan_term: int | None = None,
) -> pl.LazyFrame:
    """Select eligible orig loans before allocating quotas or sampling IDs.

    Omitted conditions impose no restriction. Term is specified in months;
    numeric comparison handles raw strings such as ``360`` and ``0360``.
    """
    if amortization_type is not None:
        if amortization_type not in ("FRM", "ARM"):
            raise ValueError("amortization_type must be FRM or ARM")
        orig = orig.filter(pl.col("amortization_type") == amortization_type)
    if original_loan_term is not None:
        if type(original_loan_term) is not int or original_loan_term <= 0:
            raise ValueError("original_loan_term must be a positive integer in months")
        orig = orig.filter(pl.col("original_loan_term").cast(pl.Int64, strict=False) == original_loan_term)
    return orig


def allocate_vintage_counts(populations: Mapping[str, int], n_loans: int) -> dict[str, int]:
    """Allocate an exact sample size proportionally using largest remainders."""
    if type(n_loans) is not int or n_loans <= 0:
        raise ValueError("n_loans must be a positive integer")
    if not populations or any(type(n) is not int or n < 0 for n in populations.values()):
        raise ValueError("Vintage populations must be nonnegative integers")
    total = sum(populations.values())
    if n_loans > total:
        raise ValueError("Requested sample exceeds the orig loan population")
    counts = {v: n_loans * populations[v] // total for v in sorted(populations)}
    order = sorted(populations, key=lambda v: (-(n_loans * populations[v] % total), v))
    for vintage in order[:n_loans - sum(counts.values())]:
        counts[vintage] += 1
    return counts


def sample_loan_ids(
    loan_ids: pl.DataFrame, *, n_loans: int, seed: int, vintage: str,
) -> pl.DataFrame:
    """Sample without replacement, independent of input row order.

    Reproducibility is defined for the same Python environment, seed, vintage,
    and set of orig identifiers. No performance or outcome fields are used.
    """
    ids = loan_ids.get_column("loan_identifier")
    if ids.dtype != pl.String:
        raise ValueError("Loan identifiers must be strings")
    if ids.null_count() or ids.str.strip_chars().eq("").any() or ids.n_unique() != len(ids):
        raise ValueError("Orig loan identifiers must be nonempty and unique")
    if type(n_loans) is not int or not 0 <= n_loans <= len(ids):
        raise ValueError("Quarter sample size must be between zero and its population")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    ordered = ids.sort().to_list()
    selected = random.Random(f"{seed}:{vintage}").sample(ordered, n_loans)
    return pl.DataFrame({"loan_identifier": sorted(selected)}, schema={"loan_identifier": pl.String})


def scan_loan_panel(
    orig_path: str | Path, perf_path: str | Path, sampled_ids: pl.DataFrame, *, vintage: str,
) -> pl.LazyFrame:
    """Join complete sampled perf histories to static fields, preserving raw codes.

    Orig-only loans have no panel rows. ``period`` is retained and ``month``
    is a Date at month start. The static join must be many-to-one.
    """
    ids = sampled_ids.select("loan_identifier").lazy()
    orig = pl.scan_parquet(orig_path, hive_partitioning=False).join(ids, on="loan_identifier", how="semi")
    perf = pl.scan_parquet(perf_path, hive_partitioning=False).join(ids, on="loan_identifier", how="semi")
    period = pl.when(pl.col("period").str.contains(r"^\d{6}$")).then(
        pl.col("period")
    ).otherwise(pl.lit("invalid"))
    return perf.join(orig, on="loan_identifier", how="inner", validate="m:1").with_columns(
        pl.lit(vintage).alias("vintage"),
        period.str.to_date("%Y%m", strict=True).alias("month"),
    )

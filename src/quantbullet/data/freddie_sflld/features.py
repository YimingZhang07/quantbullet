"""Polars cleaning, features and descriptive states for Freddie loan-month panels.

Origination month is defined here as first payment month minus one month.
It is an approximation, not a disclosed closing date. Macro lag1 columns
refer to observation months, not historical publication availability.
All reported rows remain; target definitions and modeling filters belong
to downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


# Match Zillow's full state names to Freddie's two-letter state codes.
STATE_CODES = dict(zip(
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|"
    "District of Columbia|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|"
    "Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|"
    "Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|"
    "New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|"
    "Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|"
    "Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming".split("|"),
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN "
    "MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA "
    "WV WI WY".split(), strict=True,
))

# Derived column conventions: d_ = Date, c_ = numeric, f_ = categorical String,
# is_ = Boolean. Targets (y_) belong to downstream model preparation.
# Put orig/prev/current/updated before the variable name; lag1 stays at the end
# and denotes the previous observation month.
# Leading underscores identify internal helpers omitted from the final panel.
#
# Both mappings use: raw_column -> (derived_column, missing_codes).
# Missing codes are field-specific Freddie sentinels, not valid measurements.
# Numeric features replace these codes with null before casting to Float64.
ORIG_NUMBERS = {
    "classic_fico": ("c_orig_fico", ("9999",)),
    "original_upb": ("c_orig_balance", ()),
    "original_loan_term": ("c_orig_term", ()),
    "original_interest_rate": ("c_orig_rate", ()),
    "original_ltv": ("c_orig_ltv", ("999",)),
    "original_cltv": ("c_orig_cltv", ("999",)),
    "original_dti": ("c_orig_dti", ("999",)),
}
# Factors keep the original category codes as Strings, including units and
# borrower counts; this step does not decode categories or create dummy columns.
# Example: ("f_purpose", ("9",)) names the output and treats "9" as missing.
# ("9",) is a one-item tuple; () means no additional field-specific missing codes.
# Whitespace and empty strings are handled by _text for every mapped field.
ORIG_FACTORS = {
    "loan_purpose": ("f_purpose", ("9",)),
    "occupancy_status": ("f_occupancy", ("9",)),
    "property_type": ("f_property_type", ("99",)),
    "property_state": ("f_state", ()),
    "channel": ("f_channel", ("9",)),
    "first_time_homebuyer_indicator": ("f_first_time_buyer", ("9",)),
    "number_of_units": ("f_units", ("99",)),
    "number_of_borrowers": ("f_borrowers", ("99",)),
    "vintage": ("f_vintage", ()),
}
# These lists describe panel columns, not a recommended model feature selection.
# Current-month fields and statuses may contain information about the outcome;
# downstream models must choose their inputs and timing explicitly.
NUMERIC_FEATURES = (
    "c_age", *(name for name, _ in ORIG_NUMBERS.values()),
    "c_prev_balance", "c_prev_rate", "c_orig_pmms", "c_pmms_lag1",
    "c_sato", "c_incentive", "c_orig_hpi", "c_hpi_lag1", "c_cpi_lag1",
    "c_factor", "c_hpi_growth", "c_updated_ltv", "c_current_hpi", "c_current_pmms",
)
CATEGORICAL_FEATURES = (
    *(name for name, _ in ORIG_FACTORS.values()),
    "f_month", "f_prev_modified", "f_pre_status", "f_status", "f_exit_reason",
    "f_hpi_level", "f_hpi_region_id",
)
FEATURE_COLUMNS = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
# Flags describe continuity and source inconsistencies, not model eligibility.
# is_post_exit uses complete history and is a retrospective diagnostic.
QUALITY_FLAGS = (
    "is_consecutive_month", "is_known_exit", "is_unknown_exit_code",
    "is_missing_exit_month", "is_event_month_mismatch",
    "is_zero_balance_without_exit", "is_post_exit",
)
DERIVED_COLUMNS = (
    "d_first_payment_month", "d_origination_month", "d_maturity_month",
    "d_reporting_month", "d_exit_month", *FEATURE_COLUMNS, *QUALITY_FLAGS,
)
# current_loan_delinquency_status -> descriptive state for a non-exit row.
# Numeric codes count delinquent months; group 03-99 into DQ90_PLUS.
# Normalize legacy one-digit numeric codes to two digits before lookup.
# Blank/null/unrecognized values default to UNKNOWN; exit reasons take
# precedence over this mapping when constructing f_status.
LOAN_STATUS_MAP = {
    "00": "CURRENT",
    "01": "DQ30",
    "02": "DQ60",
    **{f"{code:02d}": "DQ90_PLUS" for code in range(3, 100)},
    "RA": "REO",
    "XX": "UNKNOWN",
}

# Recognized zero_balance_code values mark termination of dataset tracking.
# Sales/securitizations can end tracking without the borrower paying off the loan.
# SFLLD Release 47 definitions (July 2026), General User Guide, Zero Balance Code:
# https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf
# 01: voluntary payoff (prepayment or maturity); 02: third-party sale;
# 03: short sale or charge-off; 09: disposal of real-estate-owned (REO) property;
# 15: whole-loan sale; 16: securitization of a reperforming loan;
# 96: confirmed underwriting or major servicing defect before a credit event.
# Keep distinct reasons and leave target definitions/filtering to the model.
# In particular, 01 remains VOLUNTARY_PAYOFF without a maturity-based split.
EXIT_REASONS = {
    "01": "VOLUNTARY_PAYOFF", "02": "THIRD_PARTY_SALE",
    "03": "SHORT_SALE_OR_CHARGE_OFF", "09": "REO_DISPOSITION",
    "15": "WHOLE_LOAN_SALE", "16": "REPERFORMING_SECURITIZATION",
    "96": "DEFECT",
}


def _text(name: str, missing: tuple[str, ...] = ()) -> pl.Expr:
    """Trim text and map blanks/supplied missing codes to null; preserve nulls."""
    return pl.col(name).str.strip_chars().replace(["", *missing], [None] * (len(missing) + 1))


def _number(name: str, missing: tuple[str, ...] = ()) -> pl.Expr:
    return _text(name, missing).cast(pl.Float64, strict=True)


def _month(name: str) -> pl.Expr:
    value = _text(name)
    return pl.when(value.is_null() | value.str.contains(r"^\d{6}$")).then(
        value
    ).otherwise(pl.lit("invalid")).str.to_date("%Y%m", strict=True)


def _positive(value: pl.Expr) -> pl.Expr:
    return pl.when(value > 0).then(value).otherwise(None)


def _require(frame: pl.LazyFrame, columns: list[str]) -> None:
    missing = set(columns) - set(frame.collect_schema().names())
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")


@dataclass(frozen=True)
class MacroTables:
    """Validated, small monthly tables used repeatedly across vintages."""

    state_hpi: pl.DataFrame
    national_hpi: pl.DataFrame
    pmms: pl.DataFrame
    cpi: pl.DataFrame


def _validate_macro(frame: pl.DataFrame, keys: list[str], table: str) -> pl.DataFrame:
    if frame.is_empty() or frame.schema["month"] != pl.Date:
        raise ValueError(f"{table}: expected a nonempty monthly Date table")
    frame = frame.with_columns(pl.col("value").cast(pl.Float64, strict=True))
    invalid = frame.select(
        pl.any_horizontal(*(pl.col(key).is_null() for key in keys)).any(),
        (pl.col("month").dt.day() != 1).any().alias("not_month_start"),
        (~pl.col("value").is_finite()).fill_null(False).any().alias("nonfinite"),
    ).row(0)
    if any(invalid):
        raise ValueError(f"{table}: missing keys, invalid months, or nonfinite values")
    if frame.select(pl.struct(keys).n_unique()).item() != frame.height:
        raise ValueError(f"{table}: duplicate macro keys")
    return frame


def prepare_macro_tables(hpi: pl.DataFrame, pmms: pl.DataFrame, cpi: pl.DataFrame) -> MacroTables:
    """Select Zillow state/national ZHVI and national PMMS/CPI; reject bad keys."""
    hpi = hpi.filter((pl.col("provider") == "zillow") & (pl.col("metric") == "ZHVI"))
    state = hpi.filter(pl.col("geography_level") == "state").select(
        pl.col("region_name").replace_strict(STATE_CODES, default=None).alias("f_state"),
        "region_id", "month", "value",
    )
    national = hpi.filter(pl.col("geography_level") == "national").select("region_id", "month", "value")
    state = _validate_macro(state, ["f_state", "month"], "state HPI")
    national = _validate_macro(national, ["month"], "national HPI")
    if national["region_id"].null_count() or national["region_id"].n_unique() != 1:
        raise ValueError("national HPI: expected one nonempty region identifier")
    if state["region_id"].null_count() or state.group_by("f_state").agg(
        pl.col("region_id").n_unique().alias("ids")
    ).filter(pl.col("ids") != 1).height:
        raise ValueError("state HPI: each state must have one nonempty region identifier")
    pmms = _validate_macro(pmms.filter(pl.col("series_id") == "MORTGAGE30US").select(
        "month", "value"
    ), ["month"], "PMMS")
    cpi = _validate_macro(cpi.filter(pl.col("series_id") == "CPIAUCNS").select(
        "month", "value"
    ), ["month"], "CPI")
    return MacroTables(
        state.with_columns(_positive(pl.col("value")).alias("value")),
        national.with_columns(_positive(pl.col("value")).alias("value")), pmms, cpi,
    )


def derive_loan_features(loans: pl.LazyFrame, *, macro: MacroTables) -> pl.LazyFrame:
    """Derive static features once per sampled loan, including origination lookups.

    Internal HPI candidates are retained for the monthly pairwise fallback;
    ``prepare_loan_months`` removes them from the published panel.
    """
    _require(loans, ["loan_identifier", "first_payment_date", "maturity_date", *ORIG_NUMBERS, *ORIG_FACTORS])
    # Unpack each mapping as raw, (name, missing), clean the source expression,
    # and alias its result to the derived name. prepare_loan_months separately
    # retains the original columns alongside these new features in the output.
    features = loans.select(
        "loan_identifier",
        _month("first_payment_date").alias("d_first_payment_month"),
        _month("maturity_date").alias("d_maturity_month"),
        *(_number(raw, missing).alias(name) for raw, (name, missing) in ORIG_NUMBERS.items()),
        *(_text(raw, missing).alias(name) for raw, (name, missing) in ORIG_FACTORS.items()),
    ).with_columns(
        pl.col("d_first_payment_month").dt.offset_by("-1mo").alias("d_origination_month"),
    )
    return features.join(macro.pmms.lazy().rename({
        "month": "d_origination_month", "value": "c_orig_pmms",
    }), on="d_origination_month", how="left", validate="m:1").join(
        macro.state_hpi.lazy().rename({
            "month": "d_origination_month", "value": "_state_orig_hpi", "region_id": "_state_region_id",
        }), on=["f_state", "d_origination_month"], how="left", validate="m:1",
    ).join(macro.national_hpi.lazy().select(
        pl.col("month").alias("d_origination_month"), pl.col("value").alias("_national_orig_hpi"),
    ), on="d_origination_month", how="left", validate="m:1").with_columns(
        (pl.col("c_orig_rate") - pl.col("c_orig_pmms")).alias("c_sato"),
    )


def _live_status() -> pl.Expr:
    raw = _text("current_loan_delinquency_status")
    code = pl.when(raw.str.contains(r"^\d{1,2}$")).then(raw.str.zfill(2)).otherwise(raw)
    return code.replace_strict(LOAN_STATUS_MAP, default="UNKNOWN")


def validate_panel(panel: pl.LazyFrame) -> int:
    """Check the small key/numeric projection before publishing a partition."""
    _require(panel, [
        "loan_identifier", "month", "current_actual_upb", "current_interest_rate",
        "current_loan_delinquency_status", "modification_flag", "zero_balance_code",
        "zero_balance_effective_date",
    ])
    if panel.collect_schema()["month"] != pl.Date:
        raise ValueError("Panel month must be Date")
    keys = panel.select("loan_identifier", "month").collect(engine="streaming")
    if keys.select(
        (pl.col("loan_identifier").is_null() | (pl.col("loan_identifier").str.strip_chars() == "")).any()
        | pl.col("month").is_null().any() | (pl.col("month").dt.day() != 1).any()
    ).item():
        raise ValueError("Panel contains missing identifiers or invalid months")
    if keys.select(pl.struct("loan_identifier", "month").n_unique()).item() != keys.height:
        raise ValueError("Panel contains duplicate loan-month keys")
    if panel.select(pl.any_horizontal(
        *[~_number(name).is_finite().fill_null(True) for name in ("current_actual_upb", "current_interest_rate")]
    ).any()).collect(engine="streaming").item():
        raise ValueError("Panel contains nonfinite numeric values")
    return keys.height


def prepare_loan_months(
    panel: pl.LazyFrame, loan_features: pl.LazyFrame, *, macro: MacroTables,
) -> pl.LazyFrame:
    """Preserve all raw rows and add features, descriptive states and quality flags.

    Input keys/numbers must be checked with ``validate_panel`` before writing.
    All lags use a sorted within-loan window and require calendar adjacency.
    """
    raw_columns = panel.collect_schema().names()
    if set(raw_columns) & set(DERIVED_COLUMNS):
        raise ValueError("Input panel already contains derived columns")
    frame = panel.join(loan_features, on="loan_identifier", how="left", validate="m:1").sort(
        "loan_identifier", "month",
    ).with_columns(
        pl.col("month").alias("d_reporting_month"),
        _number("current_actual_upb").alias("_balance"),
        _number("current_interest_rate").alias("_rate"),
        _month("zero_balance_effective_date").alias("d_exit_month"),
        _text("zero_balance_code").alias("_exit_code"),
        _live_status().alias("_live_status"),
        pl.when(_text("modification_flag").is_null()).then(pl.lit("N")).when(
            _text("modification_flag").is_in(["Y", "P"])
        ).then(pl.lit("Y")).otherwise(pl.lit("UNKNOWN")).alias("_modified"),
        pl.col("month").dt.offset_by("-1mo").alias("_macro_month"),
    ).with_columns(
        (pl.col("month").shift(1).over("loan_identifier").dt.offset_by("1mo") == pl.col("month"))
        .fill_null(False).alias("is_consecutive_month"),
        pl.col("_exit_code").is_in(list(EXIT_REASONS)).fill_null(False).alias("is_known_exit"),
        ((pl.col("month").dt.year() - pl.col("d_origination_month").dt.year()) * 12
         + pl.col("month").dt.month().cast(pl.Int32)
         - pl.col("d_origination_month").dt.month().cast(pl.Int32)).alias("c_age"),
    ).with_columns(
        pl.col("_exit_code").replace_strict(EXIT_REASONS, default=None).alias("f_exit_reason"),
        (pl.col("_exit_code").is_not_null() & ~pl.col("is_known_exit")).alias("is_unknown_exit_code"),
        (pl.col("_exit_code").is_not_null() & pl.col("d_exit_month").is_null())
        .alias("is_missing_exit_month"),
        (pl.col("_exit_code").is_not_null() & pl.col("d_exit_month").is_not_null()
         & (pl.col("d_exit_month") != pl.col("month"))).fill_null(False).alias("is_event_month_mismatch"),
        (pl.col("_exit_code").is_null() & (pl.col("_balance") <= 0)).fill_null(False)
        .alias("is_zero_balance_without_exit"),
        # Compare later rows to the earliest reported/effective known exit month.
        # This is descriptive only: keep rows and never quarantine earlier states.
        pl.when(pl.col("is_known_exit")).then(pl.min_horizontal("month", "d_exit_month"))
        .alias("_row_exit"),
        pl.when(pl.col("is_consecutive_month")).then(pl.col("_balance").shift(1).over("loan_identifier"))
        .alias("c_prev_balance"),
        pl.when(pl.col("is_consecutive_month")).then(pl.col("_rate").shift(1).over("loan_identifier"))
        .alias("c_prev_rate"),
        pl.when(pl.col("is_consecutive_month")).then(pl.col("_modified").shift(1).over("loan_identifier"))
        .alias("f_prev_modified"),
    ).with_columns(
        pl.col("_row_exit").min().over("loan_identifier").alias("_first_exit"),
    ).with_columns(
        (pl.col("month") > pl.col("_first_exit")).fill_null(False).alias("is_post_exit"),
        # Status describes this raw row. Date discrepancies remain separate flags;
        # do not move events, infer targets, or overwrite a loan's later records.
        pl.when(pl.col("is_known_exit")).then(pl.col("f_exit_reason"))
        .when(pl.col("is_unknown_exit_code")).then(pl.lit("UNKNOWN_EXIT"))
        .otherwise(pl.col("_live_status")).alias("f_status"),
        pl.col("month").dt.strftime("%m").alias("f_month"),
    ).with_columns(
        pl.when(pl.col("is_consecutive_month")).then(pl.col("f_status").shift(1).over("loan_identifier"))
        .alias("f_pre_status"),
    )
    for table, keys, rename in (
        (macro.state_hpi, ["f_state", "_macro_month"], {"month": "_macro_month", "value": "_state_lag1_hpi"}),
        (macro.national_hpi, ["_macro_month"], {"month": "_macro_month", "value": "_national_lag1_hpi"}),
        (macro.pmms, ["_macro_month"], {"month": "_macro_month", "value": "c_pmms_lag1"}),
        (macro.cpi, ["_macro_month"], {"month": "_macro_month", "value": "c_cpi_lag1"}),
        (macro.state_hpi, ["f_state", "month"], {"value": "_state_current_hpi"}),
        (macro.national_hpi, ["month"], {"value": "_national_current_hpi"}),
        (macro.pmms, ["month"], {"value": "c_current_pmms"}),
    ):
        right = table.drop("region_id", strict=False).rename(rename).lazy()
        frame = frame.join(right, on=keys, how="left", validate="m:1")
    state_pair = pl.col("_state_orig_hpi").is_not_null() & pl.col("_state_lag1_hpi").is_not_null()
    return frame.with_columns(
        pl.when(state_pair).then(pl.lit("state")).otherwise(pl.lit("national")).alias("f_hpi_level"),
        pl.when(state_pair).then(pl.col("_state_region_id")).otherwise(
            pl.lit(macro.national_hpi["region_id"][0])
        ).alias("f_hpi_region_id"),
        pl.when(state_pair).then(pl.col("_state_orig_hpi")).otherwise(pl.col("_national_orig_hpi"))
        .alias("c_orig_hpi"),
        pl.when(state_pair).then(pl.col("_state_lag1_hpi")).otherwise(pl.col("_national_lag1_hpi"))
        .alias("c_hpi_lag1"),
        pl.when(state_pair).then(pl.col("_state_current_hpi")).otherwise(pl.col("_national_current_hpi"))
        .alias("c_current_hpi"),
        (pl.col("c_prev_rate") - pl.col("c_pmms_lag1")).alias("c_incentive"),
        (pl.col("c_prev_balance") / _positive(pl.col("c_orig_balance"))).alias("c_factor"),
    ).with_columns(
        (pl.col("c_hpi_lag1") / _positive(pl.col("c_orig_hpi")) - 1).alias("c_hpi_growth"),
        (pl.col("c_orig_ltv") * pl.col("c_factor") * pl.col("c_orig_hpi")
         / _positive(pl.col("c_hpi_lag1"))).alias("c_updated_ltv"),
    ).select(*raw_columns, *DERIVED_COLUMNS)

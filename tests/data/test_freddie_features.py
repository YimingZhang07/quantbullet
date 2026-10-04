from dataclasses import replace
from datetime import date
import json
from pathlib import Path

import polars as pl
import pytest

from procs.freddie_sflld import prepare_panel as process
from quantbullet.data.freddie_sflld.features import (
    DERIVED_COLUMNS, FEATURE_COLUMNS, derive_loan_features, prepare_loan_months,
    prepare_macro_tables, validate_panel,
)
from quantbullet.data.freddie_sflld.schema import ORIG_COLUMNS, PERF_COLUMNS


def _loan(identifier="A", **updates):
    row = dict.fromkeys(ORIG_COLUMNS)
    row.update(
        loan_identifier=identifier, first_payment_date="201502", maturity_date="204501",
        classic_fico="750", original_upb="100000", original_interest_rate="4",
        original_loan_term="360", original_ltv="80", original_cltv="85", original_dti="30",
        property_state="CA", loan_purpose="P", occupancy_status="P", property_type="SF",
        channel="R", first_time_homebuyer_indicator="N", number_of_units="1", number_of_borrowers="2",
        vintage="2015Q1",
    )
    row.update(updates)
    return row


def _loans(rows):
    return pl.DataFrame(rows, schema={name: pl.String for name in [*ORIG_COLUMNS, "vintage"]})


def _panel(loans, records):
    static = {loan["loan_identifier"]: loan for loan in loans}
    rows = []
    for record in records:
        row = dict.fromkeys(PERF_COLUMNS)
        row.update(current_interest_rate="4", current_loan_delinquency_status="00", loan_age="0")
        row.update(static[record["loan_identifier"]])
        row.update(record)
        row["month"] = date(int(row["period"][:4]), int(row["period"][4:]), 1)
        rows.append(row)
    names = [*PERF_COLUMNS, *(name for name in ORIG_COLUMNS if name != "loan_identifier"), "vintage"]
    return pl.DataFrame(rows, schema={**dict.fromkeys(names, pl.String), "month": pl.Date})


@pytest.fixture
def macro_frames():
    months = [date(2015, month, 1) for month in range(1, 6)]
    hpi_rows = [
        {"provider": "zillow", "metric": "ZHVI", "geography_level": level,
         "region_id": region, "region_name": name, "month": month, "value": value}
        for level, region, name, values in (
            ("state", "CA-ZHVI", "California", [200., 210., 215., None, 230.]),
            ("national", "US-ZHVI", "United States", [100., 105., 108., 110., 120.]),
        ) for month, value in zip(months, values)
    ]
    return (
        pl.DataFrame(hpi_rows),
        pl.DataFrame({"series_id": ["MORTGAGE30US"] * 5, "month": months, "value": [4., 3., 8., 5., 6.]}),
        pl.DataFrame({"series_id": ["CPIAUCNS"] * 5, "month": months, "value": [200., 201., 202., None, 204.]}),
    )


def _prepare(loans, panel, macro_frames):
    macro = prepare_macro_tables(*macro_frames)
    static = derive_loan_features(_loans(loans).lazy(), macro=macro)
    validate_panel(panel.lazy())
    return prepare_loan_months(panel.lazy(), static, macro=macro).collect()


def test_dates_lags_and_modification_do_not_reset_age(macro_frames):
    loans = [_loan()]
    panel = _panel(loans, [
        {"loan_identifier": "A", "period": "201501", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "99000", "modification_flag": "Y"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "98000", "modification_flag": "P", "loan_age": "1"},
    ]).reverse()
    result = _prepare(loans, panel, macro_frames).sort("month")
    assert result["c_age"].to_list() == [0, 1, 2]
    assert result["loan_age"].to_list() == ["0", "0", "1"]
    assert result["d_origination_month"].to_list() == [date(2015, 1, 1)] * 3
    assert result["d_first_payment_month"].to_list() == [date(2015, 2, 1)] * 3
    assert result["d_reporting_month"].equals(result["month"])
    assert result["c_prev_balance"].to_list() == [None, 100000., 99000.]
    assert result["f_prev_modified"].to_list() == [None, "N", "Y"]
    assert result["f_pre_status"].to_list() == [None, "CURRENT", "CURRENT"]
    assert result["c_pmms_lag1"].to_list() == [None, 4., 3.]
    assert result["c_incentive"].to_list() == [None, 0., 1.]
    assert result["c_current_pmms"].to_list() == [4., 3., 8.]
    assert result["c_sato"].to_list() == [0.] * 3
    assert result["is_consecutive_month"].to_list() == [False, True, True]
    assert result["f_month"].to_list() == ["01", "02", "03"]
    assert set(result.columns) == set(panel.columns) | set(DERIVED_COLUMNS)
    assert "d_feature_month" not in result.columns and not any(name.startswith("_") for name in result.columns)
    assert "c_current_pmms" in FEATURE_COLUMNS and "f_status" in FEATURE_COLUMNS
    assert not {"y_prepay", "is_at_risk", "is_model_eligible"} & set(result.columns)
    assert result.select(panel.columns).sort("month").equals(panel.sort("month"))


def test_calendar_gap_and_loan_isolation(macro_frames):
    loans = [_loan("A"), _loan("B")]
    panel = _panel(loans, [
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000"},
        {"loan_identifier": "B", "period": "201502", "current_actual_upb": "80000"},
        {"loan_identifier": "A", "period": "201504", "current_actual_upb": "98000"},
        {"loan_identifier": "B", "period": "201503", "current_actual_upb": "79000"},
    ])
    result = _prepare(loans, panel, macro_frames)
    assert result.filter(pl.col("loan_identifier") == "A")["c_prev_balance"].to_list() == [None, None]
    assert result.filter(pl.col("loan_identifier") == "B")["c_prev_balance"].to_list() == [None, 80000.]
    assert result.filter((pl.col("loan_identifier") == "A") & (pl.col("period") == "201504"))["f_pre_status"].item() is None


def test_reference_payment_and_monthly_split(macro_frames):
    loans = [_loan()]
    panel = _panel(loans, [
        {"loan_identifier": "A", "period": "201504", "current_actual_upb": " 98000 "},
        {"loan_identifier": "A", "period": "201505", "current_actual_upb": "97000"},
    ])
    result = _prepare(loans, panel.reverse(), macro_frames).sort("month")
    payment = 100000 * (4 / 1200) / (1 - (1 + 4 / 1200) ** -360)
    first, second = result.to_dicts()
    assert result.schema["c_balance"] == pl.Float64
    assert result["c_balance"].to_list() == [98000., 97000.]
    assert result["current_actual_upb"].to_list() == [" 98000 ", "97000"]
    assert result["c_monthly_payment"].to_list() == pytest.approx([payment, payment])
    for name in ("c_interest", "c_scheduled_principal", "c_scheduled_balance"):
        assert result.schema[name] == pl.Float64 and first[name] is None
    assert second["c_interest"] == pytest.approx(98000 * 4 / 1200)
    assert second["c_scheduled_principal"] == pytest.approx(payment - second["c_interest"])
    assert second["c_scheduled_balance"] == pytest.approx(98000 - second["c_scheduled_principal"])
    assert "c_curtailment_est" not in result.columns
    assert result.select(panel.columns).equals(panel.sort("month"))


@pytest.mark.parametrize("updates, expected", [
    ({"original_interest_rate": "0"}, 100000 / 360),
    ({"original_interest_rate": None}, None),
    ({"original_interest_rate": "-1"}, None),
    ({"original_upb": None}, None),
    ({"original_upb": "0"}, None),
    ({"original_upb": "-100"}, None),
    ({"original_loan_term": None}, None),
    ({"original_loan_term": "0"}, None),
    ({"original_loan_term": "-1"}, None),
])
def test_reference_payment_zero_rate_and_invalid_inputs(macro_frames, updates, expected):
    loans = [_loan(**updates)]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "99000"},
    ]), macro_frames)
    if expected is None:
        assert result["c_monthly_payment"].null_count() == 2
        assert result["c_scheduled_principal"].null_count() == 2
        assert result["c_scheduled_balance"].null_count() == 2
    else:
        assert result["c_monthly_payment"].to_list() == pytest.approx([expected, expected])


@pytest.mark.parametrize("previous_updates", [
    {"current_actual_upb": None}, {"current_interest_rate": None},
])
def test_payment_split_missing_previous_input(macro_frames, previous_updates):
    loans = [_loan()]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000", **previous_updates},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "99000"},
    ]), macro_frames)
    assert result["c_monthly_payment"].null_count() == 0
    for name in ("c_interest", "c_scheduled_principal", "c_scheduled_balance"):
        assert result[name].null_count() == 2


@pytest.mark.parametrize("flag", ["Y", "P"])
def test_modification_flag_is_cumulative_not_retrospective(macro_frames, flag):
    loans = [_loan("A"), _loan("B", original_upb="80000")]
    panel = _panel(loans, [
        {"loan_identifier": "A", "period": "201501", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "99000",
         "modification_flag": flag, "current_interest_rate": "5"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "98000",
         "modification_flag": " ", "current_interest_rate": "6"},
        {"loan_identifier": "A", "period": "201505", "current_actual_upb": "97000"},
        {"loan_identifier": "B", "period": "201502", "current_actual_upb": "80000"},
        {"loan_identifier": "B", "period": "201503", "current_actual_upb": "79000",
         "modification_flag": "unknown"},
    ])
    result = _prepare(loans, panel.reverse(), macro_frames)
    a = result.filter(pl.col("loan_identifier") == "A").sort("month")
    b = result.filter(pl.col("loan_identifier") == "B").sort("month")
    assert result.schema["is_ever_modified"] == pl.Boolean
    assert a["is_ever_modified"].to_list() == [False, True, True, True]
    assert b["is_ever_modified"].to_list() == [False, False]
    assert a["c_monthly_payment"].n_unique() == 1
    # The modification flag/rate changes do not reset original-rate exposure.
    assert a["c_burnout"].to_list() == pytest.approx([0., 0., 0., .5])
    assert a["c_burnout_months"].to_list() == [0, 0, 0, 1]
    assert b["c_monthly_payment"][0] == pytest.approx(a["c_monthly_payment"][0] * .8)
    # Current rate changed to 6%, but March's estimate uses February's 5%.
    assert a["c_interest"][2] == pytest.approx(99000 * 5 / 1200)
    for name in ("c_interest", "c_scheduled_principal", "c_scheduled_balance"):
        assert a[name][3] is None  # April is absent, despite the persistent flag.
    assert a["c_monthly_payment"][3] is not None
    assert result.select(panel.columns).sort("loan_identifier", "month").equals(
        panel.sort("loan_identifier", "month")
    )


def test_reference_cash_flow_amounts_are_not_clipped(macro_frames):
    loans = [_loan()]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201501", "current_actual_upb": "100000",
         "current_interest_rate": "12"},
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100",
         "current_interest_rate": "0"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "0",
         "zero_balance_code": "01", "zero_balance_effective_date": "201503"},
        {"loan_identifier": "A", "period": "201504", "current_actual_upb": "0"},
    ]), macro_frames).sort("month")
    assert result.height == 4
    assert result["c_scheduled_principal"][1] < 0
    assert result["c_scheduled_balance"][2] < 0
    assert result["c_scheduled_balance"][3] < 0
    assert result["is_post_exit"][3]


def test_hpi_pair_fallback_and_updated_ltv(macro_frames):
    loans = [_loan()]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201504", "current_actual_upb": "98000"},
        {"loan_identifier": "A", "period": "201505", "current_actual_upb": "97000"},
    ]), macro_frames)
    april, may = result.sort("month").to_dicts()
    assert april["f_hpi_level"] == "state"
    assert april["c_orig_hpi"] == 200. and april["c_hpi_lag1"] == 215.
    assert april["c_current_hpi"] is None  # Diagnostic does not change the pair's geography.
    assert may["f_hpi_level"] == "national" and may["f_hpi_region_id"] == "US-ZHVI"
    assert may["c_orig_hpi"] == 100. and may["c_hpi_lag1"] == 110.
    assert may["c_current_hpi"] == 120.
    assert may["c_factor"] == pytest.approx(.98)
    assert april["c_hpi_ratio"] == pytest.approx(215 / 200)
    assert may["c_hpi_ratio"] == pytest.approx(1.1)
    assert may["c_updated_ltv"] == pytest.approx(80 * .98 * 100 / 110)
    assert may["c_cpi_lag1"] is None
    assert may["is_consecutive_month"] is True


@pytest.mark.parametrize("ratio", [1.1, 1.0, .9])
def test_hpi_ratio_appreciation_unchanged_and_depreciation(macro_frames, ratio):
    hpi, pmms, cpi = macro_frames
    hpi = hpi.with_columns(
        pl.when((pl.col("geography_level") == "state") & (pl.col("month") == date(2015, 2, 1)))
        .then(pl.lit(200 * ratio)).otherwise(pl.col("value")).alias("value"),
    )
    loans = [_loan()]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "99000"},
    ]), (hpi, pmms, cpi)).sort("month")
    assert result["c_hpi_ratio"].to_list() == pytest.approx([1., ratio])
    assert result["c_updated_ltv"][1] == pytest.approx(80 / ratio)
    assert "c_hpi_growth" not in result.columns


@pytest.mark.parametrize("month", [1, 2])
@pytest.mark.parametrize("invalid", [None, 0., -1.])
def test_hpi_ratio_invalid_pair_is_null(macro_frames, month, invalid):
    hpi, pmms, cpi = macro_frames
    hpi = hpi.with_columns(
        pl.when(pl.col("month") == date(2015, month, 1))
        .then(pl.lit(invalid, dtype=pl.Float64)).otherwise(pl.col("value")).alias("value"),
    )
    loans = [_loan()]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "99000"},
    ]), (hpi, pmms, cpi))
    assert result["f_hpi_level"][0] == "national"
    assert result["c_hpi_ratio"][0] is None


def test_unknown_state_and_official_missing_codes(macro_frames):
    loans = [_loan(property_state="PR", classic_fico="9999", original_ltv="999", original_cltv="999",
                   original_dti="999", number_of_units="99", number_of_borrowers="99",
                   first_time_homebuyer_indicator="9", property_type="99")]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "99000"},
    ]), macro_frames).sort("month")
    assert result["f_hpi_level"].to_list() == ["national"] * 2
    for name in ("c_orig_fico", "c_orig_ltv", "c_orig_cltv", "c_orig_dti", "f_units", "f_borrowers", "f_first_time_buyer", "f_property_type"):
        assert result[name].null_count() == 2
    assert result["classic_fico"].to_list() == ["9999"] * 2
    assert result["is_consecutive_month"].to_list() == [False, True]


@pytest.mark.parametrize("code,maturity,status", [
    ("01", "204501", "VOLUNTARY_PAYOFF"), ("01", "201503", "VOLUNTARY_PAYOFF"),
    ("01", "201502", "VOLUNTARY_PAYOFF"), ("01", None, "VOLUNTARY_PAYOFF"),
    ("02", "204501", "THIRD_PARTY_SALE"), ("03", "204501", "SHORT_SALE_OR_CHARGE_OFF"),
    ("09", "204501", "REO_DISPOSITION"), ("15", "204501", "WHOLE_LOAN_SALE"),
    ("16", "204501", "REPERFORMING_SECURITIZATION"), ("96", "204501", "DEFECT"),
])
def test_distinct_exit_reasons_without_target_or_maturity_split(macro_frames, code, maturity, status):
    loans = [_loan(maturity_date=maturity)]
    panel = _panel(loans, [
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "0", "zero_balance_code": code,
         "zero_balance_effective_date": "201503"},
        {"loan_identifier": "A", "period": "201504", "current_actual_upb": "0", "zero_balance_code": code,
         "zero_balance_effective_date": "201504"},
    ])
    result = _prepare(loans, panel, macro_frames).sort("month")
    assert result["f_status"].to_list() == ["CURRENT", status, status]
    assert result["f_exit_reason"].to_list() == [None, status, status]
    assert result["is_post_exit"].to_list() == [False, False, True]
    assert result["d_maturity_month"].null_count() == (3 if maturity is None else 0)
    assert result["c_prev_balance"].to_list() == [None, 100000., 0.]


@pytest.mark.parametrize("effect", ["201502", "201504", None])
def test_event_mismatch_flag_does_not_quarantine_states(macro_frames, effect):
    loans = [_loan()]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201501", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "99000"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "0", "zero_balance_code": "01",
         "zero_balance_effective_date": effect},
    ]), macro_frames).sort("month")
    assert result["f_status"].to_list() == ["CURRENT", "CURRENT", "VOLUNTARY_PAYOFF"]
    assert result["is_event_month_mismatch"].to_list() == [False, False, effect is not None]
    assert result["is_missing_exit_month"].to_list() == [False, False, effect is None]
    assert result["d_exit_month"][-1] == (date(int(effect[:4]), int(effect[4:]), 1) if effect else None)
    assert result.height == 3


@pytest.mark.parametrize("code", [None, "", "88"])
def test_unexplained_zero_and_unknown_exit_code(macro_frames, code):
    loans = [_loan()]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "0", "zero_balance_code": code},
    ]), macro_frames)
    assert result["is_zero_balance_without_exit"].to_list() == [False, code != "88"]
    assert result["is_unknown_exit_code"].to_list() == [False, code == "88"]
    assert result["f_status"].to_list() == ["CURRENT", "UNKNOWN_EXIT" if code == "88" else "CURRENT"]


@pytest.mark.parametrize("raw,status", [("00", "CURRENT"), ("01", "DQ30"),
    ("02", "DQ60"), ("04", "DQ90_PLUS"), ("99", "DQ90_PLUS"),
    ("RA", "REO"), ("XX", "UNKNOWN")])
def test_previous_status_mapping(macro_frames, raw, status):
    loans = [_loan()]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000", "current_loan_delinquency_status": raw},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "99000"},
    ]), macro_frames)
    assert result["f_pre_status"][-1] == status
    assert result.height == 2


def test_cross_year_age_and_missing_denominator(macro_frames):
    loans = [_loan(first_payment_date="201501", original_upb="0")]
    result = _prepare(loans, _panel(loans, [
        {"loan_identifier": "A", "period": "201501", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "99000"},
    ]), macro_frames)
    assert result["d_origination_month"][0] == date(2014, 12, 1)
    assert result["c_age"].to_list() == [1, 2]
    assert result["c_factor"].null_count() == 2
    assert result["c_updated_ltv"].null_count() == 2
    assert result["c_orig_hpi"].null_count() == 2


def test_post_exit_rows_and_anomalies_preserve_reported_states_and_lags(macro_frames):
    loans = [_loan()]
    panel = _panel(loans, [
        {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000"},
        {"loan_identifier": "A", "period": "201503", "current_actual_upb": "0", "zero_balance_code": "15",
         "zero_balance_effective_date": "201502"},
        {"loan_identifier": "A", "period": "201504", "current_actual_upb": "0", "current_loan_delinquency_status": "02"},
        {"loan_identifier": "A", "period": "201505", "current_actual_upb": "100", "current_loan_delinquency_status": "01"},
    ])
    result = _prepare(loans, panel, macro_frames)
    assert result.height == panel.height
    assert result["f_status"].to_list() == ["CURRENT", "WHOLE_LOAN_SALE", "DQ60", "DQ30"]
    assert result["f_pre_status"].to_list() == [None, "CURRENT", "WHOLE_LOAN_SALE", "DQ60"]
    assert result["c_prev_balance"].to_list() == [None, 100000., 0., 0.]
    assert result["is_post_exit"].to_list() == [False, True, True, True]
    assert result["is_zero_balance_without_exit"].to_list() == [False, False, True, False]
    assert result.select(panel.columns).equals(panel)


@pytest.mark.parametrize("table", [0, 1, 2])
def test_duplicate_macro_keys_fail(macro_frames, table):
    frames = list(macro_frames)
    frames[table] = pl.concat([frames[table], frames[table].head(1)])
    with pytest.raises(ValueError, match="duplicate macro"):
        prepare_macro_tables(*frames)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_macro_values_fail(macro_frames, value):
    hpi, pmms, cpi = macro_frames
    with pytest.raises(ValueError, match="nonfinite"):
        prepare_macro_tables(hpi, pmms.with_columns(pl.lit(value).alias("value")), cpi)


@pytest.mark.parametrize("value", ["invalid", "NaN", "inf"])
def test_bad_panel_numbers_fail(macro_frames, value):
    panel = _panel([_loan()], [{"loan_identifier": "A", "period": "201502", "current_actual_upb": value}])
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        validate_panel(panel.lazy())


def test_duplicate_panel_keys_fail():
    panel = _panel([_loan()], [{"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000"}] * 2)
    with pytest.raises(ValueError, match="duplicate loan-month"):
        validate_panel(panel.lazy())


@pytest.mark.parametrize("field", ["first_payment_date", "zero_balance_effective_date"])
def test_invalid_dates_fail(macro_frames, field):
    loans = [_loan(**{field: "20151"})] if field == "first_payment_date" else [_loan()]
    row = {"loan_identifier": "A", "period": "201502", "current_actual_upb": "100000"}
    if field == "zero_balance_effective_date":
        row[field] = "201513"
    with pytest.raises(pl.exceptions.PolarsError):
        _prepare(loans, _panel(loans, [row]), macro_frames)


@pytest.fixture
def config(tmp_path, macro_frames):
    sample, macro = tmp_path / "sample", tmp_path / "macro"
    sample.mkdir()
    (macro / "parquet").mkdir(parents=True)
    for name, frame in zip(("hpi", "pmms", "cpi"), macro_frames):
        frame.write_parquet(macro / f"parquet/{name}.parquet")
    loans = [_loan("A"), _loan("B", vintage="2015Q2"), _loan("C", vintage="2015Q2")]
    _loans(loans).write_parquet(sample / "sampled_loans.parquet")
    vintages = []
    for vintage, identifier, count in (("2015Q1", "A", 1), ("2015Q2", "B", 2)):
        panel = _panel(loans, [
            {"loan_identifier": identifier, "period": "201502", "current_actual_upb": "100000"},
            {"loan_identifier": identifier, "period": "201503", "current_actual_upb": "0", "zero_balance_code": "01",
             "zero_balance_effective_date": "201503"},
        ])
        directory = sample / "panel" / f"vintage={vintage}"
        directory.mkdir(parents=True)
        panel.write_parquet(directory / "panel.parquet")
        vintages.append({"vintage": vintage, "sampled_loans": count, "panel_rows": panel.height})
    (sample / "sampling_summary.json").write_text(json.dumps({"sampled_loans": 3, "vintages": vintages}))
    return process.PreparationConfig(sample, macro, tmp_path / "prepared")


def test_pipeline_and_complete_replacement(config):
    summary = process.build_prepared_panel(config)
    assert summary["rows"] == 4
    assert summary["sampled_loans"] == 3 and summary["panel_loans"] == 2
    assert summary["loans_without_perf"] == 1
    assert summary["quality_counts"]["is_known_exit"] == 2
    assert not {"model_rows", "model_prepays", "prepays", "model_features", "exclusions"} & set(summary)
    assert summary["feature_columns"] == list(FEATURE_COLUMNS)
    assert "c_hpi_ratio" in summary["feature_columns"]
    assert "c_hpi_growth" not in summary["feature_columns"]
    assert "c_hpi_ratio" in summary["vintages"][0]["feature_missing"]
    assert summary["conventions"]["hpi_ratio"] == "lag1 ZHVI divided by origination ZHVI; 1.0 means unchanged"
    assert summary["burnout_threshold"] == .5
    assert summary["vintages"][0]["feature_missing"]["c_burnout"]["rows"] == 0
    assert summary["ever_modified_rows"] == 0
    assert summary["vintages"][0]["feature_missing"]["c_interest"]["rows"] == 1
    text = (config.output_root / "preparation_summary.json").read_text()
    assert str(config.sample_root) not in text and str(config.output_root) not in text
    before = pl.read_parquet(config.output_root / "panel.parquet")
    assert before["d_reporting_month"].is_sorted()
    assert summary["path"] == "panel.parquet"
    assert summary["sorted_by"] == ["d_reporting_month"]
    assert set(p.name for p in config.output_root.iterdir()) == {"panel.parquet", "preparation_summary.json"}
    assert all("path" not in row for row in summary["vintages"])
    process.build_prepared_panel(config)
    after = pl.read_parquet(config.output_root / "panel.parquet")
    assert before.sort("loan_identifier", "month").equals(after.sort("loan_identifier", "month"))
    process.build_prepared_panel(config, vintages=["2015Q2"])
    assert not (config.output_root / "panel").exists()
    assert pl.read_parquet(config.output_root / "panel.parquet")["vintage"].unique().to_list() == ["2015Q2"]


def test_pipeline_counts_modified_rows(config):
    path = config.sample_root / "panel/vintage=2015Q1/panel.parquet"
    pl.read_parquet(path).with_columns(
        pl.when(pl.col("period") == "201503").then(pl.lit("P")).otherwise(None)
        .alias("modification_flag")
    ).write_parquet(path)
    summary = process.build_prepared_panel(config)
    assert summary["ever_modified_rows"] == 1
    assert [row["ever_modified_rows"] for row in summary["vintages"]] == [1, 0]
    assert pl.read_parquet(config.output_root / "panel.parquet")["is_ever_modified"].sum() == 1


def test_failure_preserves_successful_output(config):
    process.build_prepared_panel(config)
    summary = (config.output_root / "preparation_summary.json").read_bytes()
    original = (config.output_root / "panel.parquet").read_bytes()
    path = config.sample_root / "panel/vintage=2015Q2/panel.parquet"
    frame = pl.read_parquet(path)
    pl.concat([frame, frame.head(1)]).write_parquet(path)
    with pytest.raises(ValueError, match="duplicate loan-month"):
        process.build_prepared_panel(config)
    assert (config.output_root / "preparation_summary.json").read_bytes() == summary
    assert (config.output_root / "panel.parquet").read_bytes() == original
    assert not list(config.output_root.parent.glob(".prepared-prepare-*"))


def test_failed_publish_restores_previous_directory(config, monkeypatch):
    process.build_prepared_panel(config)
    before = (config.output_root / "preparation_summary.json").read_bytes()
    rename = Path.rename

    def fail_staging(path, target):
        if path.name.startswith(".prepared-prepare-"):
            raise PermissionError("simulated publish failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging)
    with pytest.raises(PermissionError, match="publish failure"):
        process.build_prepared_panel(config)
    assert (config.output_root / "preparation_summary.json").read_bytes() == before


def test_failed_merge_preserves_previous_output(config, monkeypatch):
    process.build_prepared_panel(config)
    before = (config.output_root / "panel.parquet").read_bytes()

    def fail_merge(paths, target):
        target.write_bytes(b"incomplete")
        raise RuntimeError("simulated merge failure")

    monkeypatch.setattr(process, "_merge_monthly", fail_merge)
    with pytest.raises(RuntimeError, match="merge failure"):
        process.build_prepared_panel(config)
    assert (config.output_root / "panel.parquet").read_bytes() == before
    assert not list(config.output_root.parent.glob(".prepared-prepare-*"))


def test_month_sorted_merge_with_many_and_empty_inputs(tmp_path):
    paths, originals = [], []
    for index, months in enumerate(([1, 3, 5], [2, 3, 4], [], [1, 5], [2])):
        frame = pl.DataFrame({
            "loan_identifier": [f"loan-{index}-{month}" for month in months],
            "d_reporting_month": [date(2015, month, 1) for month in months],
            "value": [None if month == 3 else float(month) for month in months],
        }, schema={"loan_identifier": pl.String, "d_reporting_month": pl.Date, "value": pl.Float64})
        path = tmp_path / f"quarter-{index}.parquet"
        frame.write_parquet(path)
        paths.append(path)
        originals.append(frame)
    target = tmp_path / "panel.parquet"
    process._merge_monthly(paths, target)
    output = pl.read_parquet(target)
    assert output["d_reporting_month"].is_sorted()
    assert output.sort("loan_identifier").equals(pl.concat(originals).sort("loan_identifier"))


@pytest.mark.parametrize("column,value", [("original_upb", "bad"), ("original_interest_rate", "inf"),
    ("first_payment_date", "201513"), ("maturity_date", "20151")])
def test_bad_static_values_fail_before_publication(config, column, value):
    path = config.sample_root / "sampled_loans.parquet"
    pl.read_parquet(path).with_columns(pl.lit(value).alias(column)).write_parquet(path)
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        process.build_prepared_panel(config)
    assert not config.output_root.exists()


def test_config_environment_expansion_and_missing_variable(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[data]\nsample_root="${PANEL_TEST_ROOT}/sample"\nmacro_root="macro"\noutput_root="prepared"\n')
    monkeypatch.delenv("PANEL_TEST_ROOT", raising=False)
    with pytest.raises(ValueError, match="Set environment variable PANEL_TEST_ROOT"):
        process.read_config(path)
    monkeypatch.setenv("PANEL_TEST_ROOT", str(tmp_path))
    result = process.read_config(path)
    assert result.sample_root == tmp_path / "sample"
    assert result.macro_root == tmp_path / "macro"
    assert result.output_root == tmp_path / "prepared"
    assert result.burnout_threshold == .5
    with path.open("a") as file:
        file.write('[features]\nburnout_threshold=1.0\n')
    assert process.read_config(path).burnout_threshold == 1.


@pytest.mark.parametrize("value", ['-1', 'nan', 'inf', '"0.5"', 'true'])
def test_invalid_burnout_config(tmp_path, value):
    path = tmp_path / "config.toml"
    path.write_text('[data]\nsample_root="sample"\nmacro_root="macro"\noutput_root="prepared"\n'
                    f'[features]\nburnout_threshold={value}\n')
    with pytest.raises(ValueError, match="finite nonnegative"):
        process.read_config(path)


def test_pipeline_uses_configured_burnout_threshold(config):
    # March has only January exposure. Make it exceed the default 0.5 threshold.
    path = config.macro_root / "parquet/pmms.parquet"
    pmms = pl.read_parquet(path)
    pmms.with_columns(
        pl.when(pl.col("month") == date(2015, 1, 1)).then(3.).otherwise(pl.col("value")).alias("value")
    ).write_parquet(path)
    process.build_prepared_panel(config)
    before = pl.read_parquet(config.output_root / "panel.parquet")
    assert before["c_burnout"].max() == .5
    summary = process.build_prepared_panel(replace(config, burnout_threshold=1.))
    assert summary["burnout_threshold"] == 1.
    after = pl.read_parquet(config.output_root / "panel.parquet")
    assert after["c_burnout"].max() == 0.
    assert after["c_burnout_months"].max() == 0
    assert before.drop("c_burnout", "c_burnout_months").equals(after.drop("c_burnout", "c_burnout_months"))


def test_invalid_paths_and_missing_vintage(config):
    with pytest.raises(ValueError, match="separate"):
        process.build_prepared_panel(replace(config, output_root=config.sample_root))
    with pytest.raises(ValueError, match="outside the repository"):
        process.build_prepared_panel(replace(config, output_root=Path(__file__).resolve().parents[2] / "local-data"))
    with pytest.raises(ValueError, match="must exist"):
        process.build_prepared_panel(config, vintages=["2015Q3"])
    config.output_root.mkdir()
    (config.output_root / "notes.txt").write_text("keep")
    with pytest.raises(ValueError, match="only this process"):
        process.build_prepared_panel(config)
    assert (config.output_root / "notes.txt").read_text() == "keep"


def test_orig_only_quarter_and_duplicate_sample_ids(config):
    path = config.sample_root / "panel/vintage=2015Q2/panel.parquet"
    pl.read_parquet(path).head(0).write_parquet(path)
    summary_path = config.sample_root / "sampling_summary.json"
    sampling = json.loads(summary_path.read_text())
    sampling["vintages"][1]["panel_rows"] = 0
    summary_path.write_text(json.dumps(sampling))
    summary = process.build_prepared_panel(config)
    assert summary["rows"] == 2 and summary["loans_without_perf"] == 2
    source = config.sample_root / "sampled_loans.parquet"
    frame = pl.read_parquet(source)
    pl.concat([frame, frame.head(1)]).write_parquet(source)
    with pytest.raises(ValueError, match="globally unique"):
        process.build_prepared_panel(config)
    assert json.loads((config.output_root / "preparation_summary.json").read_text())["rows"] == 2

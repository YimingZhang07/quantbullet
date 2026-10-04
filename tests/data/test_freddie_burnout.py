"""Calendar-based burnout: history boundaries, missing data and loan isolation."""

from datetime import date

import polars as pl
import pytest

from quantbullet.data.freddie_sflld import add_burnout_features


def _panel(months, *, identifier="A", orig=date(2015, 1, 1), rate=5.):
    return pl.DataFrame({
        "loan_identifier": [identifier] * len(months),
        "d_origination_month": [orig] * len(months),
        "c_orig_rate": [rate] * len(months),
        "d_reporting_month": [date(2015, month, 1) for month in months],
        "raw": [f"record-{month}" for month in months],
    }, schema={"loan_identifier": pl.String, "d_origination_month": pl.Date,
               "c_orig_rate": pl.Float64, "d_reporting_month": pl.Date, "raw": pl.String})


@pytest.fixture
def pmms():
    return pl.DataFrame({
        "month": [date(2015, month, 1) for month in range(1, 7)],
        "value": [4.2, 4.5, 3.8, 4.7, 5.2, 6.],
    })


def test_calendar_sum_threshold_boundary_and_empty_history(pmms):
    source = _panel([1, 2, 3, 4, 5, 6])
    result = add_burnout_features(source.reverse().lazy(), pmms=pmms).collect().sort("d_reporting_month")
    assert result["c_burnout"].to_list() == pytest.approx([0., 0., .3, .3, 1., 1.])
    assert result["c_burnout_months"].to_list() == [0, 0, 1, 1, 2, 2]
    assert result.schema["c_burnout"] == pl.Float64
    assert result.schema["c_burnout_months"] == pl.Int32
    assert result.select(source.columns).equals(source)
    assert set(result.columns) - set(source.columns) == {"c_burnout", "c_burnout_months"}


def test_late_first_reporting_and_perf_gaps_do_not_lose_exposure(pmms):
    full = add_burnout_features(_panel([1, 2, 3, 4, 5, 6]).lazy(), pmms=pmms).collect()
    for months in ([6], [3, 6]):
        result = add_burnout_features(_panel(months).lazy(), pmms=pmms).collect()
        expected = full.filter(pl.col("d_reporting_month").dt.month().is_in(months))
        assert result.sort("d_reporting_month").equals(expected.sort("d_reporting_month"))
        assert result.height == len(months)


def test_no_later_macro_or_reporting_rows_change_prior_burnout(pmms):
    source = _panel([5, 6])
    baseline = add_burnout_features(source.lazy(), pmms=pmms).collect()
    later_pmms = pmms.with_columns(
        pl.when(pl.col("month") >= date(2015, 4, 1)).then(0.).otherwise(pl.col("value")).alias("value")
    )
    result = add_burnout_features(source.lazy(), pmms=later_pmms).collect()
    assert result.row(0, named=True) == baseline.row(0, named=True)  # May excludes April onward.
    shorter = add_burnout_features(source.head(1).lazy(), pmms=pmms).collect()
    assert shorter.equals(baseline.head(1))
    changed_cutoff = pmms.with_columns(
        pl.when(pl.col("month") == date(2015, 3, 1)).then(3.).otherwise(pl.col("value")).alias("value")
    )
    changed = add_burnout_features(source.lazy(), pmms=changed_cutoff).collect()
    assert changed["c_burnout"][0] == pytest.approx(1.8)


@pytest.mark.parametrize("missing", ["absent", "null", "outside_coverage"])
def test_missing_macro_makes_cumulative_exposure_unknown(pmms, missing):
    if missing == "absent":
        pmms = pmms.filter(pl.col("month") != date(2015, 3, 1))
    elif missing == "null":
        pmms = pmms.with_columns(
            pl.when(pl.col("month") == date(2015, 3, 1)).then(None).otherwise(pl.col("value")).alias("value")
        )
    else:
        pmms = pmms.filter(pl.col("month") < date(2015, 3, 1))
    result = add_burnout_features(_panel([1, 3, 5, 6]).lazy(), pmms=pmms).collect()
    assert result["c_burnout"].to_list()[:2] == pytest.approx([0., .3])
    assert result["c_burnout"].to_list()[2:] == [None, None]
    assert result["c_burnout_months"].to_list() == [0, 1, None, None]


def test_missing_macro_before_origination_is_irrelevant(pmms):
    source = _panel([5], orig=date(2015, 3, 1))
    result = add_burnout_features(source.lazy(), pmms=pmms.tail(4)).collect()
    assert result["c_burnout"].item() == pytest.approx(.7)
    assert result["c_burnout_months"].item() == 1


@pytest.mark.parametrize("updates", [
    {"orig": None}, {"rate": None}, {"rate": -1.}, {"orig": date(2015, 7, 1)},
])
def test_invalid_inputs_are_null_even_without_history(pmms, updates):
    result = add_burnout_features(_panel([1, 6], **updates).lazy(), pmms=pmms).collect()
    assert result["c_burnout"].to_list() == [None, None]
    assert result["c_burnout_months"].to_list() == [None, None]


def test_loan_isolation_and_direct_month_by_month_reference(pmms):
    source = pl.concat([_panel([3, 5, 6]), _panel([1, 6], identifier="B", rate=4.)]).reverse()
    result = add_burnout_features(source.lazy(), pmms=pmms).collect()
    for row in result.iter_rows(named=True):
        cutoff = row["d_reporting_month"].month - 2
        incentives = [row["c_orig_rate"] - value for month, value in pmms.iter_rows()
                      if row["d_origination_month"] <= month and month.month <= cutoff]
        assert row["c_burnout"] == pytest.approx(sum(max(value - .5, 0) for value in incentives))
        assert row["c_burnout_months"] == sum(value > .5 for value in incentives)


@pytest.mark.parametrize("threshold", [-.1, float("nan"), float("inf"), "0.5", True])
def test_invalid_threshold_fails(pmms, threshold):
    with pytest.raises(ValueError, match="finite nonnegative"):
        add_burnout_features(_panel([6]).lazy(), pmms=pmms, threshold=threshold)


def test_zero_threshold_and_empty_panel(pmms):
    result = add_burnout_features(_panel([6]).lazy(), pmms=pmms, threshold=0).collect()
    assert result["c_burnout"].item() == pytest.approx(2.8)
    assert result["c_burnout_months"].item() == 4
    empty = add_burnout_features(_panel([]).lazy(), pmms=pmms).collect()
    assert empty.is_empty() and empty.schema == result.schema

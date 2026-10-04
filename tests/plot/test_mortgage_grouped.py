"""Mortgage methods use the grouped system directly, with post-mean CPR."""
from datetime import date
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import polars as pl
import pytest

from quantbullet.plot.grouped_data import BinSpec, summarize_grouped_means
from quantbullet.linear_product_model.mortgage_diagnostics import MortgageColnames, MortgageDiagnostics


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_round_ties_and_step_have_distinct_membership_and_input_parity():
    records = {'x': [-.75, -.25, .25, .75, 1., None], 'y': [1., 2., 3., 4., 5., 6.]}
    options = dict(x='x', y='y', bins={'x': BinSpec.round(.5)})
    a = summarize_grouped_means(pd.DataFrame(records), **options)
    b = summarize_grouped_means(pl.DataFrame(records), **options)
    assert_frame_equal(a.summary, b.summary)
    assert a.x_positions.tolist() == [-1., 0., 1.]
    assert a.summary['count'].tolist() == [1, 2, 2]
    expected = pl.DataFrame(records).select((pl.col('x')/.5).round()*.5)['x'].drop_nulls().to_list()
    assert sorted(expected) == [-1., 0., 0., 1., 1.]
    step = summarize_grouped_means(pl.DataFrame(records), x='x', y='y', bins={'x': BinSpec.step(.5)})
    assert step.summary['count'].tolist() != a.summary['count'].tolist()


def test_decimal_rounding_uses_polars_arithmetic_for_both_inputs():
    records = {'x': [.15, .35, .95], 'y': [1., 2., 3.]}
    expected = pl.DataFrame(records).select((pl.col('x')/.1).round()*.1)['x'].to_list()
    for frame in [pd.DataFrame(records), pl.DataFrame(records)]:
        result = summarize_grouped_means(frame, x='x', y='y', bins={'x':BinSpec.round(.1)})
        np.testing.assert_allclose(result.x_positions, expected)
        assert result.summary['count'].tolist() == [1,1,1]


def test_mortgage_direct_new_chain_means_support_counts_and_cpr(monkeypatch):
    import quantbullet.plot.binned_plots as legacy
    def forbidden(*args, **kwargs):
        raise AssertionError('MortgageDiagnostics must not use the legacy entry point')
    monkeypatch.setattr(legacy, 'plot_binned_actual_vs_pred', forbidden)
    monkeypatch.setattr(legacy, 'prepare_binned_data_polars', forbidden)
    df = pl.DataFrame({'inc': [0., 0., 1., 0., 2.], 'actual': [0., 1., 0., 1., 0.],
                       'pred': [.1, .3, .2, .4, .5], 'w': [1., 3., 2., 2., 1.], 'purpose': [0, 0, 0, 1, 1]})
    diag = MortgageDiagnostics(df, MortgageColnames(response='actual', model_preds={'Model':'pred'},
                               incentive='inc', weight='w'), bin_config={'incentive': 1.}, y_transform='smm_to_cpr')
    fig, axes = diag.incentive_plot(facet_col='purpose', min_count=2, n_cols=2)
    assert len(axes) == 2 and len(fig.axes) == 4
    assert axes[0].lines[0].get_ydata()[0] == pytest.approx(1-(1-.75)**12)
    assert axes[0].lines[1].get_ydata()[0] == pytest.approx(1-(1-.25)**12)
    assert np.isnan(axes[0].lines[0].get_ydata()[1:]).all()
    assert np.isnan(axes[1].lines[0].get_ydata()).all()
    twins = [ax for ax in fig.axes if ax not in axes]
    assert [p.get_height() for p in twins[0].patches] == [2, 1, 0]
    assert [p.get_height() for p in twins[1].patches] == [1, 0, 1]
    assert twins[0].get_ylim() == twins[1].get_ylim()
    assert all(line.get_markersize() == 3 for ax in axes for line in ax.lines)
    assert 'Count (right axis)' in [t.get_text() for t in fig.legends[0].texts]


def test_pandas_ordered_categories_dates_empty_and_vintage():
    df = pd.DataFrame({'dt': [date(2025,1,1), date(2025,3,1)], 'orig': [date(2020,1,1)]*2,
                       'y': [.1,.2], 'category': pd.Categorical(['B','A'], categories=['B','A','C'], ordered=True)})
    diag = MortgageDiagnostics(df, MortgageColnames(response='y', factor_dt='dt', age='category', orig_dt='orig'),
                               bin_config={'factor_dt':'discrete', 'age':'discrete'}, y_as_percent=False)
    _, axes = diag.factor_date_plot(min_count=0)
    assert np.diff(axes[0].lines[0].get_xdata())[0] == 59
    _, axes = diag.age_plot(min_count=0)
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ['B','A','C']
    assert np.isnan(axes[0].lines[0].get_ydata()[2])
    _, axes = diag.by_vintage_year('age', min_count=0)
    assert '2020' in axes[0].get_title()
    empty = summarize_grouped_means(pl.DataFrame({'x': [], 'y': []}, schema={'x':pl.Float64, 'y':pl.Float64}), x='x', y='y')
    assert empty.summary.empty


def test_role_bin_travels_with_the_column_and_can_be_overridden():
    df = pl.DataFrame({'inc': [0., 0., 1., 0., 2.], 'actual': [0., 1., 0., 1., 0.],
                       'pred': [.1, .3, .2, .4, .5], 'w': [1., 3., 2., 2., 1.]})
    names = MortgageColnames(response='actual', model_preds={'Model': 'pred'},
                             incentive=('inc', 1.), weight='w')
    assert names.incentive == 'inc' and names.bins == {'incentive': 1.}
    diag = MortgageDiagnostics(df, names, y_as_percent=False)
    _, axes = diag.incentive_plot(min_count=0)
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), [0., 1., 2.])
    overridden = MortgageDiagnostics(df, names, bin_config={'incentive': 2.}, y_as_percent=False)
    _, axes = overridden.incentive_plot(min_count=0)
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), [0., 2.])
    _, axes = diag.plot('incentive', bins=2.)
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), [0., 2.])


def test_generic_plot_accepts_source_columns_and_bin_overrides():
    df = pl.DataFrame({'bal': [1e5, 1.4e5, 2.6e5, 3e5], 'y': [0., 1., 0., 1.], 'p': [.1, .2, .3, .4]})
    diag = MortgageDiagnostics(df, MortgageColnames(response='y', model_preds={'Model': 'p'}),
                               bin_config={'bal': 1e5}, y_as_percent=False)
    fig, axes = diag.plot('bal')
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), [1e5, 3e5])
    assert [t.get_text() for t in fig.legends[0].texts][:2] == ['Actual', 'Model']
    fig, axes = diag.plot('bal', bins=BinSpec.edges([0, 2e5, 4e5]))
    assert [p.get_height() for p in fig.axes[1].patches] == [2, 2]
    with pytest.raises(ValueError, match='neither'):
        diag.plot('missing')
    with pytest.raises(ValueError, match='incentive'):
        diag.plot('incentive')
    with pytest.warns(DeprecationWarning, match='min_size'):
        diag.plot('bal', min_size=10)


def test_source_imports_do_not_reference_legacy():
    import inspect
    from quantbullet.plot import grouped_data, grouped_means
    from quantbullet.linear_product_model import mortgage_diagnostics
    for module in [grouped_data, grouped_means, mortgage_diagnostics]:
        assert 'binned_plots' not in inspect.getsource(module)
    subprocess.run([sys.executable, '-c', '''
import sys
from quantbullet.linear_product_model.mortgage_diagnostics import MortgageDiagnostics
assert 'quantbullet.plot.binned_plots' not in sys.modules
'''], check=True, capture_output=True, text=True)

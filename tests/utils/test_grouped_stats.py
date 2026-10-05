"""Grouped statistics can be used without importing a plotting library."""
import subprocess
import sys

import polars as pl
import pytest

from quantbullet.utils.grouped_stats import grouped_weighted_summary


def test_metric_specific_validity_and_row_counts():
    df = pl.DataFrame({"g": [0]*7, "a": [1., 3., None, 5., 7., float('inf'), 9.],
                       "b": [2., None, 4., 8., 10., 12., 14.],
                       "w": [1., 3., 2., None, 0., 1., float('inf')]})
    result = grouped_weighted_summary(df, by=['g'], metrics=['a', 'b'], weight='w').row(0, named=True)
    assert result['count'] == 7
    assert result['weight_sum'] == 7  # every finite weight, with or without a valid metric
    assert result['a__valid_count'] == 3
    assert result['a__weight_sum'] == 4
    assert result['a__weighted_sum'] == 10
    assert result['a__mean'] == 2.5
    assert result['b__valid_count'] == 4
    assert result['b__mean'] == pytest.approx(22/4)


def test_equal_weights_zero_denominator_null_keys_and_empty():
    df = pl.DataFrame({'g': [None, None, 'A'], 'y': [2., 4., None], 'w': [0., 0., 1.]})
    unweighted = grouped_weighted_summary(df, by=['g'], metrics=['y'])
    assert unweighted['y__mean'].to_list() == [3., None]
    assert unweighted['weight_sum'].to_list() == [2., 1.]  # unit weights match count
    weighted = grouped_weighted_summary(df, by=['g'], metrics=['y'], weight='w')
    assert weighted['y__mean'].to_list() == [None, None]
    assert weighted['y__valid_count'].to_list() == [2, 0]
    empty = grouped_weighted_summary(df.head(0), by=['g'], metrics=['y'], weight='w')
    assert empty.height == 0 and empty.schema['y__mean'] == pl.Float64


@pytest.mark.parametrize('weight', [-1., float('-inf')])
def test_negative_weight_raises_even_if_metric_missing(weight):
    with pytest.raises(ValueError, match='nonnegative'):
        grouped_weighted_summary(pl.DataFrame({'g': [1], 'y': [None], 'w': [weight]}),
                                 by=['g'], metrics=['y'], weight='w')


def test_import_without_plotting_dependencies():
    code = '''
import sys
import polars as pl
from quantbullet.utils.grouped_stats import grouped_weighted_summary
assert not any(name.startswith(('matplotlib', 'quantbullet.plot')) for name in sys.modules)
assert grouped_weighted_summary(pl.DataFrame({'g':[1], 'y':[2.]}), by=['g'], metrics=['y'])['y__mean'][0] == 2
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

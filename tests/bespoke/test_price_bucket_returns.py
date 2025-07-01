import unittest
import numpy as np
import pandas as pd
import logging
import statsmodels.api as sm
from quantbullet.utils.data import generate_fake_loan_prices
from quantbullet.dfutils.label import get_bins_and_labels
from quantbullet.log_config import setup_logger

class TestPriceBucketReturns(unittest.TestCase):
    def setUp( self ):
        self.data = generate_fake_loan_prices(
            start_date   = '2023-01-01',
            end_date     = '2023-12-31',
            price_range  = (50, 105),
            n_tickers    = 100,
            overlap_pct  = 0.8,
            seed         = 42
        )

    def test_price_bucket_returns( self ):
        df = self.data.copy()
        # this sorting is necessary to ensure that the shift operation works correctly
        df = df.sort_values(by=['LoanID', 'Date']).reset_index(drop=True)
        df['Price_t']      = df['Price']
        df['Balance_t']    = df['Balance']
        df['Date_t']       = df['Date']

        # by doing this, we avoid merging data; the shift is applied within each LoanID group
        # and the data is assigned back using the same index
        df['Price_t+1']    = df.groupby('LoanID')['Price'].shift(-1)
        df['Balance_t+1']  = df.groupby('LoanID')['Balance'].shift(-1)
        df['Date_t+1']     = df.groupby('LoanID')['Date'].shift(-1)

        df['Price_diff']   = df['Price_t+1'] - df['Price_t']
        df['Price_return'] = df['Price_diff'] / df['Price_t']
        df['Date_diff']    = (df['Date_t+1'] - df['Date_t']).dt.days

        cutoffs = [ 70, 80, 90, 100 ]
        cutoff_bins, cutoff_labels = get_bins_and_labels(cutoffs, include_inf=True)

        df['Price_bucket_t'] = pd.cut(
            df['Price_t'],
            bins=cutoff_bins,
            labels=cutoff_labels,
            include_lowest=True
        )

        # pick a bucket and conduct the analysis
        bucket = '<70'
        bucket_df = df[ df['Price_bucket_t'] == bucket ]

        def wavg( group, value_col, weight_col):
            return np.sum(group[value_col] * group[weight_col]) / np.sum(group[weight_col])
        

        # A loan may not have consecutive prices, so we may get NaN values in Price_t+1.
        # if we don't drop these rows, the weighted average will count those rows in the denominator, but not in the numerator,
        # which will lead to incorrect results.
        bucket_df = bucket_df.dropna(subset=['Price_t+1'])

        # after the group by, we get a Series with the Date as index. The series has no name by default...
        # If we turn this Series into a DataFrame, the column will be '0' by default.

        diff = ( bucket_df.groupby('Date')
            .apply(lambda x: wavg(x, 'Price_diff', 'Balance_t'), include_groups=False) )

        diff.name = 'Wavg_price_diff'

        returns = ( bucket_df.groupby('Date')
            .apply(lambda x: wavg(x, 'Price_return', 'Balance_t'), include_groups=False) )

        returns.name = 'Wavg_price_return'

        # Optional: create a fake index and do the regression analysis

        np.random.seed(42)  # Set a fixed random seed for reproducibility
        index = pd.Series(
            index = diff.index,
            data = np.random.randn(len(diff)),  # Use a fixed random state for reproducibility
            name = 'Index'
        )

        x = sm.add_constant(index)
        model = sm.OLS(returns, x).fit()
        self.assertAlmostEqual(model.params.loc[ 'Index' ], -0.001, places=2)
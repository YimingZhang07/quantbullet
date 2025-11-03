import unittest
import numpy as np
import pandas as pd
from quantbullet.linear_product_model.mtg_perf_eval import MtgModelPerformanceEvaluator, MtgPerfColnames

def create_test_data() -> pd.DataFrame:
    np.random.seed(42)

    # parameters
    n_loans = 5000
    loan_ids = [f"L{i:05d}" for i in range(n_loans)]

    # origination dates between 2020 and 2025
    orig_dates = pd.to_datetime(
        np.random.choice(pd.date_range("2020-01-01", "2025-01-01", freq="W"), size=n_loans)
    )

    # choose factor date after origination (loan age in months)
    ages = np.random.randint(1, 60, size=n_loans)
    factor_dates = orig_dates + pd.to_timedelta(ages * 30, unit="D")

    # loan purpose
    purposes = np.random.choice(["Refi", "Cashout"], size=n_loans, p=[0.6, 0.4])

    # CLTV: tighter for refi, higher for cashout
    cltv = np.random.uniform(20, 80, size=n_loans)

    # rate incentive (negative means below market)
    rate_incentive = np.random.uniform(-2, 2, n_loans)

    # balance
    balances = np.random.uniform(5000, 500000, n_loans)

    # prepayment rate dynamic — driven by incentive, age, and cltv
    base_prepay = 0.05 + 0.02 * np.random.randn(n_loans)
    prepay_rate = (
        base_prepay
        + 0.10 * rate_incentive * 10  # more incentive → higher prepay
        + 0.02 * (100 - cltv)         # lower cltv → more mobility
        + 0.05 * ages             # seasoning effect
    )

    # rescale the prepayment rates to be between 0 to 5% based on prepay_rate
    prepay_rate = (prepay_rate - prepay_rate.min()) / (prepay_rate.max() - prepay_rate.min()) * 0.05

    # assemble
    df = pd.DataFrame({
        "loan_id": loan_ids,
        "origination_date": orig_dates,
        "factor_date": factor_dates,
        "age": ages,
        "loan_purpose": purposes,
        "cltv": cltv.round(3),
        "incentive": rate_incentive.round(3),
        "balance": balances.round(2),
        "prepay_rate": prepay_rate.round(4),
        "model_pred" : np.random.uniform(0, 0.05, n_loans).round(4)
    })

    # derived realism — ensure factor_date <= 2025-12-31
    df = df[df["factor_date"] <= "2025-12-31"].reset_index(drop=True)
    return df


class TestMTGPerfEval(unittest.TestCase):
    def setUp(self):
        pass

    def test_incentive_plots(self):
        """Test incentive by vintage year plots generation."""
        colnames = MtgPerfColnames(
            incentive       ='incentive',
            dt              ='factor_date',
            orig_dt         ='origination_date',
            response        ='prepay_rate',
            model_preds={
                'Model' : 'model_pred'
            },
            weighted_by='balance'
        )
        df = create_test_data()
        # add some large numbers to vintage 2023 for testing
        df.loc[df['origination_date'].dt.year == 2023, 'balance'] += 10e6
        evaluator = MtgModelPerformanceEvaluator(
            df=df,
            colname_mapping=colnames
        )
        evaluator.incentive_by_vintage_year_plots(n_quantile_bins=20, n_cols=3, scatter_size_by='sum_weights', scatter_size_scale=0.3)
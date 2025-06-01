import pandas as pd
import numpy as np
import random
from datetime import date, timedelta
from typing import Union

def download_fama_french_five_factors_daily(start, end):
    """
    Download the Fama-French Five-Factor model data from the Ken French data library
    
    Parameters
    ----------
    start : datetime.date
        The start date for the data
    end : datetime.date
        The end date for the data
        
    Returns
    -------
    pandas.DataFrame
        The Fama-French Five Factors
    """
    # Import the pandas_datareader package
    import pandas_datareader.data as web

    # Use DataReader to fetch the Fama-French Five-Factor model data
    ff_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_Daily', 'famafrench', start, end)

    # The Fama-French dataset can contain different tables. For example, '0' is typically the annual data,
    # '1' might be the monthly data, etc. Here we access the daily data.
    ff_daily = ff_factors[0]

    return ff_daily


# def generate_fake_bond_trades( start_date: str | date,
#                                end_date: str | date,
#                                freq: str = "D") -> pd.DataFrame:
#     """
#     Generate a DataFrame of fake bond trades for testing purposes.
    
#     The DataFrame will contain the following columns:
#     - date: The date of the trade
#     - ticker: The ticker symbol of the bond
#     - rating: The rating of the bond (e.g., AAA, AA, A, etc.)
#     - feature_A: A random feature value
#     - feature_B: A random feature value
#     - feature_C: A random feature value
#     - feature_D: A random feature value
#     - expiry: The expiry date of the bond
#     - yield: The yield of the bond
    
#     Returns
#     -------
#     pandas.DataFrame
#         A DataFrame containing the fake bond trades.
#     """
#     base_tickers = [f"TICK{i:03}" for i in range(1, 51)]
#     ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
#     date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

#     records = []
#     for current_date in date_range:
#         for rating in ratings:
#             # For each rating, randomly pick 10–20 tickers to simulate trading on that day
#             traded_today = random.sample(base_tickers, k=random.randint(10, 20))
#             for ticker in traded_today:
#                 expiry_date = current_date + timedelta(days=random.randint(30, 3650))
#                 means = [5, 20, 50, 100]
#                 std_devs = [1, 5, 10, 20]
#                 features = [np.random.normal(mean, std_dev) for mean, std_dev in zip(means, std_devs)]
#                 yield_value = features[0] * 0.2 + features[1] * 0.2 + np.random.normal(0, 0.2)
#                 # Ensure yield_value is non-negative
#                 yield_value = max(0, yield_value)
#                 records.append({
#                     'date': current_date,
#                     'ticker': f"{ticker} {rating}",
#                     'rating': rating,
#                     'feature_A': features[0],
#                     'feature_B': features[1],
#                     'feature_C': features[2],
#                     'feature_D': features[3],
#                     'expiry': expiry_date,
#                     'yield': yield_value
#                 })

#     df_large = pd.DataFrame(records)
#     return df_large

def generate_fake_bond_trades(
    start_date: Union[str, date],
    end_date: Union[str, date],
    freq: str = "D",
    num_clusters: int = 4,
    cluster_std: float = 2.0,
    cluster_by_rating: bool = False,
    random_state: int = None,
    theo_coefficients: list = (0.15, 0.25, 0, 0),
    response_std: float = 0.3
) -> pd.DataFrame:
    """
    Generate a DataFrame of fake bond trades with cluster-friendly structure.

    Parameters
    ----------
    start_date : str or date
        Start date for the trade simulation.
    end_date : str or date
        End date for the trade simulation.
    freq : str
        Frequency of trading days (e.g., "D" for daily).
    num_clusters : int
        Number of clusters in (feature_A, feature_B) space.
    cluster_std : float
        Spread around each cluster center.
    cluster_by_rating : bool
        If True, assign cluster centers based on bond rating.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing fake bond trade data.
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    base_tickers = [f"TICK{i:03}" for i in range(1, 51)]
    ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    if cluster_by_rating:
        rating_cluster_map = {
            'AAA'   : (10, 20),
            'AA'    : (20, 30),
            'A'     : (30, 40),
            'BBB'   : (40, 50),
            'BB'    : (50, 60),
            'B'     : (60, 70)
        }
    else:
        cluster_centers = [
            (np.random.uniform(0, 100), np.random.uniform(0, 100))
            for _ in range(num_clusters)
        ]

    records = []
    for current_date in date_range:
        for rating in ratings:
            tickers_today = random.sample(base_tickers, k=random.randint(10, 20))
            for ticker in tickers_today:
                expiry_date = current_date + timedelta(days=random.randint(30, 3650))

                if cluster_by_rating:
                    center = rating_cluster_map[rating]
                else:
                    center = random.choice(cluster_centers)

                feature_A = np.random.normal(center[0], cluster_std)
                feature_B = np.random.normal(center[1], cluster_std)
                feature_C = np.random.normal(0, 10)
                feature_D = np.random.normal(0, 10)
                yield_value = (
                    theo_coefficients[0] * feature_A +
                    theo_coefficients[1] * feature_B +
                    theo_coefficients[2] * feature_C +
                    theo_coefficients[3] * feature_D +
                    np.random.normal(0, response_std)
                )
                yield_value = max(0, yield_value)

                records.append({
                    'date': current_date,
                    'ticker': f"{ticker} {rating}",
                    'rating': rating,
                    'feature_A': feature_A,
                    'feature_B': feature_B,
                    'feature_C': feature_C,
                    'feature_D': feature_D,
                    'expiry': expiry_date,
                    'yield': yield_value
                })

    return pd.DataFrame(records)
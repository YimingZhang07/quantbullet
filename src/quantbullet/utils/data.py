import datetime
import pandas as pd
import numpy as np
import random
from datetime import timedelta, date

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


def generate_fake_bond_trades( start_date: str | date,
                               end_date: str | date,
                               freq: str = "D") -> pd.DataFrame:
    """
    Generate a DataFrame of fake bond trades for testing purposes.
    
    The DataFrame will contain the following columns:
    - date: The date of the trade
    - ticker: The ticker symbol of the bond
    - rating: The rating of the bond (e.g., AAA, AA, A, etc.)
    - feature_A: A random feature value
    - feature_B: A random feature value
    - feature_C: A random feature value
    - feature_D: A random feature value
    - expiry: The expiry date of the bond
    - yield: The yield of the bond
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the fake bond trades.
    """
    base_tickers = [f"TICK{i:03}" for i in range(1, 51)]
    ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    records = []
    for current_date in date_range:
        # For each rating, randomly pick 10–20 tickers to simulate trading on that day
        for rating in ratings:
            eligible_tickers = [ticker for ticker in base_tickers]
            traded_today = random.sample(eligible_tickers, k=random.randint(10, 20))
            for ticker in traded_today:
                expiry_date = current_date + timedelta(days=random.randint(30, 3650))
                features = np.random.normal(loc=0, scale=1, size=4)
                yield_value = 5 + features[0]*1 + features[1]*2 + np.random.normal(0, 0.5)
                records.append({
                    'date': current_date,
                    'ticker': f"{ticker} {rating}",
                    'rating': rating,
                    'feature_A': features[0],
                    'feature_B': features[1],
                    'feature_C': features[2],
                    'feature_D': features[3],
                    'expiry': expiry_date,
                    'yield': yield_value
                })

    df_large = pd.DataFrame(records)
    return df_large

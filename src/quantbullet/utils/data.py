import pandas as pd
import numpy as np
import random
from datetime import date, timedelta
from typing import Union

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

def generate_fake_loan_prices(
    start_date: str,
    end_date: str,
    price_range: tuple = (90, 105),
    n_tickers: int = 100,
    overlap_pct: float = 0.8,
    seed: int = 42,
):
    """
    Generate a DataFrame of fake loan price data for testing purposes.
    """
    np.random.seed(seed)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Universe of possible tickers: LoanID0000 to LoanID9999
    all_possible_ids = set(range(10000))

    # Randomly initialize base tickers
    initial_ids = np.random.choice(list(all_possible_ids), n_tickers, replace=False)
    previous_ids = list(initial_ids)

    data = []

    for date in all_dates:
        n_overlap = int(len(previous_ids) * overlap_pct)
        n_new = len(previous_ids) - n_overlap

        # Choose overlap IDs from previous day
        overlapping_ids = np.random.choice(previous_ids, n_overlap, replace=False).tolist()

        # Exclude already used IDs to avoid duplication
        used_ids = set(previous_ids)
        available_ids = list(all_possible_ids - used_ids)
        new_ids = np.random.choice(available_ids, n_new, replace=False).tolist()

        today_ids = overlapping_ids + new_ids
        today_prices = np.random.uniform(price_range[0], price_range[1], len(today_ids))

        for loan_id, price in zip(today_ids, today_prices):
            ticker = f"LoanID{str(loan_id).zfill(4)}"
            balance = loan_id * 1000
            data.append({
                'Date': date,
                'LoanID': ticker,
                'Price': round(price, 2),
                'Balance': balance
            })

        previous_ids = today_ids

    return pd.DataFrame(data)

def generate_fake_happiness_data(
    n_samples: int = 200,
    random_state: int = 42,
):
    """
    Generate a DataFrame of fake happiness data for testing purposes.
    """
    np.random.seed(random_state)

    # Create features
    age = np.random.uniform(20, 80, n_samples)
    income = np.random.uniform(20000, 120000, n_samples)
    education = np.random.uniform(8, 20, n_samples)
    level = np.random.choice(['highschool', 'bachelor', 'master', 'phd'], n_samples)

    # Create target with non-linear relationships
    happiness = (
        0.5 * np.sin((age - 40) / 10) +  # non-linear relationship with age
        0.3 * np.log(income / 30000) +   # log relationship with income
        0.2 * education +                # linear relationship with education
        0.1 * (level == 'phd').astype(float) +  # categorical effect
        np.random.normal(0, 0.5, n_samples)  # noise
    )

    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'level': level,
        'happiness': happiness
    })

    return data
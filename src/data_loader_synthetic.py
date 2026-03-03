"""
Synthetic Data Generator (Fallback)
===================================

Generates synthetic market data for testing when real data is not available.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_synthetic_ivv_data(
    start_date: str = "2000-06-05",
    end_date: str = "2025-04-09",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic IVV-like data with realistic properties.
    
    Parameters
    ----------
    start_date : str
        Start date
    end_date : str
        End date
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Synthetic price data
    """
    np.random.seed(random_seed)
    
    # Create date range (trading days only)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    # Generate returns with regime changes
    # - Normal period: 8% annual return, 15% vol
    # - Crisis periods: -20% annual return, 40% vol
    
    returns = np.zeros(n_days)
    
    # Define regimes (approximate dates of major market events)
    # 2000-2002: Dot-com crash
    # 2008-2009: Financial crisis
    # 2020: COVID crash
    # 2022: Bear market
    
    for i in range(n_days):
        date = dates[i]
        
        # Determine regime
        if (date.year >= 2000 and date.year <= 2002):
            # Dot-com crash
            mean_ret = -0.10 / 252
            vol = 0.30 / np.sqrt(252)
        elif (date.year == 2008 or date.year == 2009):
            # Financial crisis
            mean_ret = -0.20 / 252
            vol = 0.40 / np.sqrt(252)
        elif (date.year == 2020 and date.month <= 6):
            # COVID crash
            mean_ret = -0.15 / 252
            vol = 0.35 / np.sqrt(252)
        elif (date.year == 2022):
            # 2022 bear market
            mean_ret = -0.15 / 252
            vol = 0.25 / np.sqrt(252)
        else:
            # Normal market
            mean_ret = 0.10 / 252
            vol = 0.17 / np.sqrt(252)
        
        returns[i] = np.random.normal(mean_ret, vol)
    
    # Convert returns to prices
    prices = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    print(f"Generated synthetic IVV data: {len(df)} days from {df.index[0]} to {df.index[-1]}")
    
    return df


def generate_synthetic_fed_funds_rate(
    start_date: str = "2000-06-05",
    end_date: str = "2025-04-09",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic Federal Funds rate data.
    
    Parameters
    ----------
    start_date : str
        Start date
    end_date : str
        End date
    random_seed : int
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Synthetic Fed Funds rate data
    """
    np.random.seed(random_seed + 1)
    
    # Create date range (all days, not just trading days)
    dates = pd.date_range(start=start_date, end=end_date)
    n_days = len(dates)
    
    # Generate rates with regime changes
    rates = np.zeros(n_days)
    
    for i in range(n_days):
        date = dates[i]
        
        # Approximate historical Fed Funds rate levels
        if date.year <= 2000:
            rate = 6.0 + np.random.randn() * 0.2
        elif date.year <= 2003:
            rate = 2.0 + np.random.randn() * 0.3
        elif date.year <= 2007:
            rate = 4.0 + np.random.randn() * 0.5
        elif date.year <= 2015:
            rate = 0.25 + np.random.randn() * 0.1
        elif date.year <= 2019:
            rate = 2.0 + np.random.randn() * 0.3
        elif date.year <= 2021:
            rate = 0.10 + np.random.randn() * 0.05
        elif date.year <= 2023:
            rate = 3.5 + np.random.randn() * 0.5
        else:
            rate = 4.5 + np.random.randn() * 0.3
        
        rates[i] = max(0.01, rate)  # Floor at 0.01%
    
    df = pd.DataFrame({
        'rate': rates
    }, index=dates)
    
    print(f"Generated synthetic Fed Funds data: {len(df)} days from {df.index[0]} to {df.index[-1]}")
    
    return df


def get_synthetic_experiment_data() -> Tuple[pd.DataFrame, int]:
    """
    Generate synthetic data for experiments.
    
    Returns
    -------
    Tuple[pd.DataFrame, int]
        (data, n_days) where data contains returns and rf rates
    """
    print("\n" + "=" * 80)
    print("USING SYNTHETIC DATA (MASSIVE_TOKEN not set)")
    print("=" * 80)
    
    from .data_loader import prepare_data
    
    # Generate synthetic data
    ivv_df = generate_synthetic_ivv_data()
    ff_df = generate_synthetic_fed_funds_rate()
    
    # Prepare data
    data = prepare_data(ivv_df, ff_df, strategy_start_date="2000-06-06")
    n_days = len(data)
    
    return data, n_days

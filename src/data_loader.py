"""
Data Loading Utilities
======================

Functions to load and prepare market data for volatility scaling experiments.
Uses the massive API for financial data retrieval.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple
from massive import RESTClient


def load_ivv_data(start_date: str = "2000-06-05", end_date: str = "2025-04-09") -> pd.DataFrame:
    """
    Load IVV (iShares Core S&P 500 ETF) daily adjusted close prices.
    
    Parameters
    ----------
    start_date : str
        Start date in format 'YYYY-MM-DD' (one day before strategy start to compute first return)
    end_date : str
        End date in format 'YYYY-MM-DD'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, close
        Index: date (datetime)
    """
    api_key = os.getenv("MASSIVE_TOKEN")
    if not api_key:
        raise ValueError("MASSIVE_TOKEN environment variable not set")
    
    client = RESTClient(api_key=api_key)
    
    # Fetch aggregates (daily bars)
    aggs = []
    for agg in client.list_aggs(
        ticker="IVV",
        multiplier=1,
        timespan="day",
        from_=start_date,
        to=end_date,
        limit=50000
    ):
        aggs.append(agg)
    
    if not aggs:
        raise ValueError(f"No data retrieved for IVV from {start_date} to {end_date}")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': pd.Timestamp(agg.timestamp, unit='ms'),
        'open': agg.open,
        'high': agg.high,
        'low': agg.low,
        'close': agg.close,
        'volume': agg.volume
    } for agg in aggs])
    
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')
    
    print(f"Loaded {len(df)} days of IVV data from {df.index[0]} to {df.index[-1]}")
    
    return df[['close']]


def load_fed_funds_rate(start_date: str = "2000-06-05", end_date: str = "2025-04-09") -> pd.DataFrame:
    """
    Load Federal Funds Effective Rate from FRED.
    
    Parameters
    ----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, rate (annualized percent)
        Index: date (datetime)
    """
    import pandas_datareader as pdr
    
    # Load from FRED
    df = pdr.data.DataReader('DFF', 'fred', start_date, end_date)
    df.columns = ['rate']
    df.index.name = 'date'
    
    print(f"Loaded {len(df)} days of Fed Funds rate from {df.index[0]} to {df.index[-1]}")
    
    return df


def prepare_data(ivv_df: pd.DataFrame, ff_df: pd.DataFrame, 
                 strategy_start_date: str = "2000-06-06") -> pd.DataFrame:
    """
    Prepare aligned data for backtesting.
    
    Computes:
    - Simple returns from IVV close prices
    - Daily risk-free rate from Fed Funds rate
    - Aligns both series to IVV trading days
    
    Parameters
    ----------
    ivv_df : pd.DataFrame
        IVV price data with 'close' column
    ff_df : pd.DataFrame
        Fed Funds rate data with 'rate' column (annualized percent)
    strategy_start_date : str
        First trading day for strategy (returns computed from prior day)
    
    Returns
    -------
    pd.DataFrame
        Aligned data with columns: close, return, rf_rate, rf_return
        Index: date (datetime)
    """
    # Compute simple returns: r_k = P_k / P_{k-1} - 1
    df = ivv_df.copy()
    df['return'] = df['close'].pct_change()
    
    # Align Fed Funds rate to IVV trading days (forward-fill missing values)
    df = df.join(ff_df, how='left')
    df['rate'] = df['rate'].ffill()
    
    # Count forward-fills
    n_fills = df['rate'].isna().sum()
    if n_fills > 0:
        print(f"Warning: {n_fills} Fed Funds rate values were missing and forward-filled")
    
    # Fill any remaining NaNs at the start with the first valid value
    df['rate'] = df['rate'].bfill()
    
    # Convert annualized percent to daily simple return
    # Paper uses money-market convention: r_rf_k = (rate_k / 100) / 360
    df['rf_return_360'] = (df['rate'] / 100.0) / 360.0
    
    # Also compute using 252 day convention as diagnostic
    df['rf_return_252'] = (df['rate'] / 100.0) / 252.0
    
    # Default to 360 convention (money-market)
    df['rf_return'] = df['rf_return_360']
    
    # Filter to strategy start date onwards
    df = df[df.index >= strategy_start_date].copy()
    
    # Drop first row if return is NaN
    df = df.dropna(subset=['return'])
    
    print(f"Prepared data: {len(df)} trading days from {df.index[0]} to {df.index[-1]}")
    print(f"Average daily return: {df['return'].mean():.6f}")
    print(f"Average daily rf (360): {df['rf_return_360'].mean():.6f}")
    print(f"Average daily rf (252): {df['rf_return_252'].mean():.6f}")
    
    return df


def get_experiment_data() -> Tuple[pd.DataFrame, int]:
    """
    Load and prepare all data needed for experiments.
    
    Returns
    -------
    Tuple[pd.DataFrame, int]
        (data, n_days) where data contains returns and rf rates,
        n_days is the total number of trading days
    """
    # Check if MASSIVE_TOKEN is available
    api_key = os.getenv("MASSIVE_TOKEN")
    
    if not api_key:
        print("\n" + "=" * 80)
        print("WARNING: MASSIVE_TOKEN not set. Using synthetic data for demonstration.")
        print("To use real data, set the MASSIVE_TOKEN environment variable.")
        print("=" * 80)
        
        # Use synthetic data
        from .data_loader_synthetic import get_synthetic_experiment_data
        return get_synthetic_experiment_data()
    
    # Load real data
    ivv_df = load_ivv_data(start_date="2000-06-05", end_date="2025-04-09")
    ff_df = load_fed_funds_rate(start_date="2000-06-05", end_date="2025-04-09")
    
    # Prepare aligned data
    data = prepare_data(ivv_df, ff_df, strategy_start_date="2000-06-06")
    
    n_days = len(data)
    
    return data, n_days

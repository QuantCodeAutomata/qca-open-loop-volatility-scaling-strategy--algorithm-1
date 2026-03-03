"""
Performance Metrics
==================

Functions to compute performance metrics for volatility scaling strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_volatility_tracking_error(
    index_returns: np.ndarray,
    halflife: int = 126,
    target_vol_annual: float = 0.15,
    trading_days: int = 252
) -> float:
    """
    Compute volatility tracking error.
    
    For each day k, compute EWMA vol of index returns sigma_hat_k^ind.
    Annualized tracking error = mean over all k of |sigma_hat_k^ind * sqrt(252) - 0.15|
    
    Parameters
    ----------
    index_returns : np.ndarray
        Strategy index returns r_k^ind
    halflife : int
        EWMA halflife in days
    target_vol_annual : float
        Target annualized volatility (default: 0.15 = 15%)
    trading_days : int
        Trading days per year for annualization
    
    Returns
    -------
    float
        Annualized volatility tracking error
    """
    from .ewma import compute_ewma_vol
    
    # Compute EWMA volatility of index returns
    sigma_hat_ind = compute_ewma_vol(index_returns, halflife)
    
    # Annualize
    sigma_hat_ind_annual = sigma_hat_ind * np.sqrt(trading_days)
    
    # Tracking error: mean absolute deviation from target
    tracking_error = np.mean(np.abs(sigma_hat_ind_annual - target_vol_annual))
    
    return tracking_error


def compute_annualized_return(
    index_levels: np.ndarray,
    trading_days: int = 252
) -> float:
    """
    Compute annualized return (CAGR).
    
    CAGR = (R_N)^(252/N) - 1
    
    Parameters
    ----------
    index_levels : np.ndarray
        Cumulative index levels R_k^ind (starting from R_0 = 1)
    trading_days : int
        Trading days per year
    
    Returns
    -------
    float
        Annualized return
    """
    N = len(index_levels) - 1  # Number of returns (excluding R_0)
    R_N = index_levels[-1]
    
    if R_N <= 0:
        return -1.0
    
    cagr = R_N ** (trading_days / N) - 1
    
    return cagr


def compute_annualized_volatility(
    index_returns: np.ndarray,
    trading_days: int = 252
) -> float:
    """
    Compute annualized volatility.
    
    Ann. Vol = std(r_1^ind,...,r_N^ind) * sqrt(252)
    
    Parameters
    ----------
    index_returns : np.ndarray
        Strategy index returns
    trading_days : int
        Trading days per year
    
    Returns
    -------
    float
        Annualized volatility
    """
    return np.std(index_returns, ddof=1) * np.sqrt(trading_days)


def compute_sharpe_ratio(
    annualized_return: float,
    annualized_volatility: float
) -> float:
    """
    Compute Sharpe ratio.
    
    Sharpe Ratio = Annualized Return / Annualized Volatility
    (no risk-free subtraction as per paper methodology)
    
    Parameters
    ----------
    annualized_return : float
        Annualized return
    annualized_volatility : float
        Annualized volatility
    
    Returns
    -------
    float
        Sharpe ratio
    """
    if annualized_volatility == 0:
        return 0.0
    
    return annualized_return / annualized_volatility


def compute_maximum_drawdown(index_levels: np.ndarray) -> float:
    """
    Compute maximum drawdown.
    
    Max DD = max over all k of (max_{j<=k} R_j^ind - R_k^ind) / max_{j<=k} R_j^ind
    
    Parameters
    ----------
    index_levels : np.ndarray
        Cumulative index levels R_k^ind
    
    Returns
    -------
    float
        Maximum drawdown (as positive number, e.g., 0.387 for 38.7%)
    """
    running_max = np.maximum.accumulate(index_levels)
    drawdowns = (running_max - index_levels) / running_max
    max_dd = np.max(drawdowns)
    
    return max_dd


def compute_all_metrics(
    index_returns: np.ndarray,
    index_levels: np.ndarray,
    halflife: int = 126,
    target_vol_annual: float = 0.15,
    trading_days: int = 252
) -> Dict[str, float]:
    """
    Compute all five Table 1 performance metrics.
    
    Parameters
    ----------
    index_returns : np.ndarray
        Strategy index returns r_k^ind
    index_levels : np.ndarray
        Cumulative index levels R_k^ind (including R_0 = 1)
    halflife : int
        EWMA halflife for tracking error computation
    target_vol_annual : float
        Target annualized volatility
    trading_days : int
        Trading days per year
    
    Returns
    -------
    Dict[str, float]
        Dictionary with keys: tracking_error, ann_return, ann_vol, sharpe, max_dd
    """
    tracking_error = compute_volatility_tracking_error(
        index_returns, halflife, target_vol_annual, trading_days
    )
    
    ann_return = compute_annualized_return(index_levels, trading_days)
    ann_vol = compute_annualized_volatility(index_returns, trading_days)
    sharpe = compute_sharpe_ratio(ann_return, ann_vol)
    max_dd = compute_maximum_drawdown(index_levels)
    
    return {
        'tracking_error': tracking_error,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd
    }


def format_metrics(metrics: Dict[str, float], strategy_name: str) -> str:
    """
    Format metrics as a human-readable string.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Metrics dictionary
    strategy_name : str
        Name of the strategy
    
    Returns
    -------
    str
        Formatted metrics string
    """
    lines = [
        f"\n{strategy_name} Performance Metrics:",
        f"  Volatility Tracking Error: {metrics['tracking_error']:.4f} ({metrics['tracking_error']*100:.2f}%)",
        f"  Annualized Return:         {metrics['ann_return']:.4f} ({metrics['ann_return']*100:.2f}%)",
        f"  Annualized Volatility:     {metrics['ann_vol']:.4f} ({metrics['ann_vol']*100:.2f}%)",
        f"  Sharpe Ratio:              {metrics['sharpe']:.4f}",
        f"  Maximum Drawdown:          {metrics['max_dd']:.4f} ({metrics['max_dd']*100:.2f}%)"
    ]
    return '\n'.join(lines)

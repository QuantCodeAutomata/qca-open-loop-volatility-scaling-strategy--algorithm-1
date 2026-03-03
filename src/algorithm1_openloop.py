"""
Algorithm 1: Open-Loop Volatility Scaling
=========================================

Implements the open-loop volatility scaling strategy from the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict
from .ewma import compute_ewma_vol


def run_openloop_strategy(
    risky_returns: np.ndarray,
    rf_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    halflife: int = 126,
    target_vol_annual: float = 0.15,
    max_leverage: float = 1.5,
    trading_days: int = 252
) -> Dict[str, np.ndarray]:
    """
    Run the open-loop volatility scaling strategy (Algorithm 1).
    
    At each day k:
      Step A - Realize returns:
        r_k^ind = r_k * w_{k-1} + r_rf_k * (1 - w_{k-1})
        R_k^ind = R_{k-1}^ind * (1 + r_k^ind)
      Step B - Update EWMA volatility of risky asset:
        sigma_hat_k = EWMA(r_1,...,r_k) with h=126
      Step C - Decide new weight:
        sigma_tar = 0.15 / sqrt(252)  [daily target volatility]
        L = 1.5  [leverage limit]
        w_k = min(sigma_tar / sigma_hat_k, L)
        c_k = 1 - w_k
    
    Parameters
    ----------
    risky_returns : np.ndarray
        Risky asset returns r_k (IVV)
    rf_returns : np.ndarray
        Risk-free returns r_rf_k
    dates : pd.DatetimeIndex
        Trading dates
    halflife : int
        EWMA halflife (default: 126)
    target_vol_annual : float
        Target annualized volatility (default: 0.15)
    max_leverage : float
        Maximum leverage (default: 1.5)
    trading_days : int
        Trading days per year (default: 252)
    
    Returns
    -------
    Dict[str, np.ndarray]
        Results dictionary with keys:
        - dates: trading dates
        - risky_returns: r_k
        - rf_returns: r_rf_k
        - weights: w_k (risky asset weight)
        - cash_weights: c_k (cash weight)
        - index_returns: r_k^ind
        - index_levels: R_k^ind (including R_0 = 1)
        - risky_vol: sigma_hat_k (daily)
        - index_vol: sigma_hat_k^ind (daily)
    """
    N = len(risky_returns)
    
    # Daily target volatility
    sigma_tar = target_vol_annual / np.sqrt(trading_days)
    
    # Compute beta for EWMA
    beta = np.exp(-np.log(2) / halflife)
    
    # Initialize arrays
    weights = np.zeros(N)
    cash_weights = np.zeros(N)
    index_returns = np.zeros(N)
    index_levels = np.zeros(N + 1)
    risky_vol = np.zeros(N)
    index_vol = np.zeros(N)
    
    # Initial conditions
    w_prev = 1.0  # w_0 = 1 (fully invested)
    index_levels[0] = 1.0  # R_0 = 1
    
    # EWMA state variables
    raw_risky = 0.0
    raw_index = 0.0
    
    # Main loop
    for k in range(N):
        # Step A: Realize returns using previous weight
        r_ind_k = risky_returns[k] * w_prev + rf_returns[k] * (1 - w_prev)
        index_returns[k] = r_ind_k
        index_levels[k + 1] = index_levels[k] * (1 + r_ind_k)
        
        # Step B: Update EWMA volatility estimates
        # Risky asset EWMA
        raw_risky = beta * raw_risky + (1 - beta) * risky_returns[k]**2
        correction_risky = 1 - beta**(k + 1)
        sigma_hat_k = np.sqrt(raw_risky / correction_risky)
        risky_vol[k] = sigma_hat_k
        
        # Index EWMA
        raw_index = beta * raw_index + (1 - beta) * r_ind_k**2
        correction_index = 1 - beta**(k + 1)
        sigma_hat_k_ind = np.sqrt(raw_index / correction_index)
        index_vol[k] = sigma_hat_k_ind
        
        # Step C: Decide new weight (open-loop)
        w_k = min(sigma_tar / sigma_hat_k, max_leverage)
        c_k = 1 - w_k
        
        weights[k] = w_k
        cash_weights[k] = c_k
        
        # Update for next iteration
        w_prev = w_k
    
    # Return results
    results = {
        'dates': dates,
        'risky_returns': risky_returns,
        'rf_returns': rf_returns,
        'weights': weights,
        'cash_weights': cash_weights,
        'index_returns': index_returns,
        'index_levels': index_levels,
        'risky_vol': risky_vol,
        'index_vol': index_vol
    }
    
    return results


def run_buyhold_baseline(
    risky_returns: np.ndarray,
    rf_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    halflife: int = 126
) -> Dict[str, np.ndarray]:
    """
    Run IVV buy-and-hold baseline for comparison.
    
    This is equivalent to w_k = 1, c_k = 0 for all k.
    
    Parameters
    ----------
    risky_returns : np.ndarray
        Risky asset returns r_k (IVV)
    rf_returns : np.ndarray
        Risk-free returns r_rf_k (not used, but kept for consistency)
    dates : pd.DatetimeIndex
        Trading dates
    halflife : int
        EWMA halflife for volatility computation
    
    Returns
    -------
    Dict[str, np.ndarray]
        Results dictionary (same structure as run_openloop_strategy)
    """
    N = len(risky_returns)
    beta = np.exp(-np.log(2) / halflife)
    
    # For buy-and-hold: w_k = 1, c_k = 0
    weights = np.ones(N)
    cash_weights = np.zeros(N)
    
    # Index returns = risky returns (since w=1)
    index_returns = risky_returns.copy()
    
    # Compute cumulative levels
    index_levels = np.zeros(N + 1)
    index_levels[0] = 1.0
    for k in range(N):
        index_levels[k + 1] = index_levels[k] * (1 + index_returns[k])
    
    # Compute EWMA volatility
    risky_vol = compute_ewma_vol(risky_returns, halflife)
    index_vol = risky_vol.copy()  # Same as risky vol for buy-and-hold
    
    results = {
        'dates': dates,
        'risky_returns': risky_returns,
        'rf_returns': rf_returns,
        'weights': weights,
        'cash_weights': cash_weights,
        'index_returns': index_returns,
        'index_levels': index_levels,
        'risky_vol': risky_vol,
        'index_vol': index_vol
    }
    
    return results


def validate_openloop_implementation() -> bool:
    """
    Validate open-loop algorithm implementation with known test cases.
    
    Returns
    -------
    bool
        True if all validations pass
    """
    # Test 1: Verify timing - r_k^ind uses w_{k-1}
    r = np.array([0.01, 0.02, -0.01])
    rf = np.array([0.0001, 0.0001, 0.0001])
    dates = pd.date_range('2020-01-01', periods=3)
    
    results = run_openloop_strategy(r, rf, dates)
    
    # First return should use w_0 = 1.0
    expected_r0_ind = r[0] * 1.0 + rf[0] * 0.0
    assert np.isclose(results['index_returns'][0], expected_r0_ind), \
        f"Expected {expected_r0_ind}, got {results['index_returns'][0]}"
    
    # Test 2: Verify weights are bounded
    assert np.all(results['weights'] >= 0), "Weights should be non-negative"
    assert np.all(results['weights'] <= 1.5), "Weights should not exceed max leverage"
    
    # Test 3: Verify cash weights
    assert np.allclose(results['weights'] + results['cash_weights'], 1.0), \
        "Weights and cash weights should sum to 1"
    
    # Test 4: Verify R_0 = 1
    assert results['index_levels'][0] == 1.0, "Initial index level should be 1"
    
    print("✓ Open-loop algorithm validation passed")
    return True

"""
Algorithm 2: Proportional Feedback Volatility Control
=====================================================

Implements the proportional feedback controller with log-volatility tracking error.
"""

import numpy as np
import pandas as pd
from typing import Dict


def run_feedback_control_strategy(
    risky_returns: np.ndarray,
    rf_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    halflife: int = 126,
    target_vol_annual: float = 0.15,
    max_leverage: float = 1.5,
    gain: float = 50.0,
    kappa_min: float = -1.0,
    kappa_max: float = 1.0,
    theta: float = 0.5,
    warm_start_days: int = 10,
    trading_days: int = 252
) -> Dict[str, np.ndarray]:
    """
    Run the proportional feedback volatility control strategy (Algorithm 2).
    
    Per-day loop (k = 1, 2, ..., N):
      Step A - Realize returns:
        r_k^ind = r_k * w_{k-1} + r_rf_k * (1 - w_{k-1})
        R_k^ind = R_{k-1}^ind * (1 + r_k^ind)
      Step B - Update EWMA estimates:
        Update sigma_hat_k (risky asset EWMA vol using r_k)
        Update sigma_hat_k^ind (index EWMA vol using r_k^ind)
      Step C - Compute tracking error:
        e_k = log(sigma_hat_k^ind / sigma_tar)
      Step D - Update controller and set weights:
        IF k <= 10 (warm start):
          kappa_k = 0
          w_k = min(sigma_tar / sigma_hat_k, L)
        ELSE:
          clipped_input = clip(-g * e_k, kappa_min, kappa_max)
          kappa_k = (1 - theta) * clipped_input + theta * kappa_{k-1}
          w_k = min(exp(kappa_k) * sigma_tar / sigma_hat_k, L)
    
    Parameters
    ----------
    risky_returns : np.ndarray
        Risky asset returns r_k
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
    gain : float
        Proportional gain (default: 50)
    kappa_min : float
        Minimum controller state (default: -1)
    kappa_max : float
        Maximum controller state (default: 1)
    theta : float
        Smoothing parameter (default: 0.5)
    warm_start_days : int
        Warm start period (default: 10)
    trading_days : int
        Trading days per year (default: 252)
    
    Returns
    -------
    Dict[str, np.ndarray]
        Results dictionary with additional fields:
        - tracking_error: e_k
        - kappa: kappa_k (controller state)
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
    tracking_error = np.zeros(N)
    kappa = np.zeros(N)
    
    # Initial conditions
    w_prev = 1.0  # w_0 = 1
    index_levels[0] = 1.0  # R_0 = 1
    kappa_prev = 0.0  # kappa_0 = 0
    
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
        
        # Step C: Compute tracking error
        e_k = np.log(sigma_hat_k_ind / sigma_tar)
        tracking_error[k] = e_k
        
        # Step D: Update controller and set weights
        if k < warm_start_days:
            # Warm start: use open-loop weights
            kappa_k = 0.0
            w_k = min(sigma_tar / sigma_hat_k, max_leverage)
        else:
            # Engage feedback control
            clipped_input = np.clip(-gain * e_k, kappa_min, kappa_max)
            kappa_k = (1 - theta) * clipped_input + theta * kappa_prev
            w_k = min(np.exp(kappa_k) * sigma_tar / sigma_hat_k, max_leverage)
        
        c_k = 1 - w_k
        
        weights[k] = w_k
        cash_weights[k] = c_k
        kappa[k] = kappa_k
        
        # Update for next iteration
        w_prev = w_k
        kappa_prev = kappa_k
    
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
        'index_vol': index_vol,
        'tracking_error': tracking_error,
        'kappa': kappa
    }
    
    return results


def validate_feedback_implementation() -> bool:
    """
    Validate feedback control algorithm implementation.
    
    Returns
    -------
    bool
        True if all validations pass
    """
    # Test 1: Warm start period
    r = np.ones(20) * 0.01
    rf = np.ones(20) * 0.0001
    dates = pd.date_range('2020-01-01', periods=20)
    
    results = run_feedback_control_strategy(r, rf, dates, warm_start_days=10)
    
    # During warm start, kappa should be 0
    assert np.all(results['kappa'][:10] == 0.0), \
        "Kappa should be 0 during warm start"
    
    # Test 2: Verify weights are bounded
    assert np.all(results['weights'] >= 0), "Weights should be non-negative"
    assert np.all(results['weights'] <= 1.5), "Weights should not exceed max leverage"
    
    # Test 3: Verify kappa bounds (approximately)
    # After warm start, kappa should be roughly bounded by [kappa_min, kappa_max]
    assert np.all(results['kappa'] >= -1.5), "Kappa should be roughly >= -1.5"
    assert np.all(results['kappa'] <= 1.5), "Kappa should be roughly <= 1.5"
    
    # Test 4: Verify cash weights
    assert np.allclose(results['weights'] + results['cash_weights'], 1.0), \
        "Weights and cash weights should sum to 1"
    
    # Test 5: Verify R_0 = 1
    assert results['index_levels'][0] == 1.0, "Initial index level should be 1"
    
    # Test 6: Verify tracking error is log ratio
    k = 15  # After warm start
    expected_e = np.log(results['index_vol'][k] / (0.15 / np.sqrt(252)))
    assert np.isclose(results['tracking_error'][k], expected_e, rtol=1e-6), \
        f"Tracking error mismatch at k={k}"
    
    print("✓ Feedback control algorithm validation passed")
    return True

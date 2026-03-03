"""
Algorithm 3: Leverage-Limiting Feedback Control
===============================================

Implements combined volatility and leverage control strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict


def run_leverage_control_strategy(
    risky_returns: np.ndarray,
    rf_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    halflife: int = 126,
    halflife_short: int = 42,
    target_vol_annual: float = 0.15,
    max_leverage: float = 1.5,
    gain: float = 50.0,
    gain_lev: float = 20.0,
    kappa_min: float = -1.0,
    kappa_max: float = 1.0,
    kappa_min_lev: float = -2.0,
    theta: float = 0.5,
    warm_start_days: int = 10,
    trading_days: int = 252
) -> Dict[str, np.ndarray]:
    """
    Run the leverage-limiting feedback control strategy (Algorithm 3).
    
    Per-day loop (k = 1, 2, ..., N):
      Step A - Realize returns:
        r_k^ind = r_k * w_{k-1} + r_rf_k * (1 - w_{k-1})
        R_k^ind = R_{k-1}^ind * (1 + r_k^ind)
      Step B - Update EWMA estimates:
        Update sigma_hat_k (risky asset)
        Update sigma_hat_k^ind (index)
        Update MA_k^long and MA_k^short (leverage MAs)
      Step C - Compute tracking error:
        e_k = log(sigma_hat_k^ind / sigma_tar)
      Step D - Update controllers and set weights:
        IF k <= 10 (warm start):
          kappa_k = 0, kappa_k^lev = 0
          w_k = min(sigma_tar / sigma_hat_k, L)
        ELSE:
          # Volatility control
          kappa_k = (1-theta)*clip(-g*e_k, kappa_min, kappa_max) + theta*kappa_{k-1}
          # Leverage control
          kappa_k^lev = clip(-g_ell*log(MA_k^long/MA_k^short), kappa_min_lev, 0)
          # Combined weight
          w_k = min(exp(kappa_k)*sigma_tar/sigma_hat_k, exp(kappa_k^lev)*L)
    
    Parameters
    ----------
    risky_returns : np.ndarray
        Risky asset returns r_k
    rf_returns : np.ndarray
        Risk-free returns r_rf_k
    dates : pd.DatetimeIndex
        Trading dates
    halflife : int
        EWMA halflife for volatility (default: 126)
    halflife_short : int
        Short EWMA halflife for leverage control (default: 42)
    target_vol_annual : float
        Target annualized volatility (default: 0.15)
    max_leverage : float
        Maximum leverage (default: 1.5)
    gain : float
        Volatility control gain (default: 50)
    gain_lev : float
        Leverage control gain (default: 20)
    kappa_min : float
        Min volatility controller state (default: -1)
    kappa_max : float
        Max volatility controller state (default: 1)
    kappa_min_lev : float
        Min leverage controller state (default: -2)
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
        - kappa: kappa_k (volatility controller state)
        - kappa_lev: kappa_k^lev (leverage controller state)
        - ma_long: MA_k^long
        - ma_short: MA_k^short
    """
    N = len(risky_returns)
    
    # Daily target volatility
    sigma_tar = target_vol_annual / np.sqrt(trading_days)
    
    # Compute betas for EWMA
    beta = np.exp(-np.log(2) / halflife)
    beta_long = beta  # 126 days for long MA
    beta_short = np.exp(-np.log(2) / halflife_short)  # 42 days for short MA
    
    # Initialize arrays
    weights = np.zeros(N)
    cash_weights = np.zeros(N)
    index_returns = np.zeros(N)
    index_levels = np.zeros(N + 1)
    risky_vol = np.zeros(N)
    index_vol = np.zeros(N)
    tracking_error = np.zeros(N)
    kappa = np.zeros(N)
    kappa_lev = np.zeros(N)
    ma_long = np.zeros(N)
    ma_short = np.zeros(N)
    
    # Initial conditions
    w_prev = 1.0  # w_0 = 1
    index_levels[0] = 1.0  # R_0 = 1
    kappa_prev = 0.0  # kappa_0 = 0
    
    # EWMA state variables
    raw_risky = 0.0
    raw_index = 0.0
    raw_long = 0.0
    raw_short = 0.0
    
    # Main loop
    for k in range(N):
        # Step A: Realize returns using previous weight
        r_ind_k = risky_returns[k] * w_prev + rf_returns[k] * (1 - w_prev)
        index_returns[k] = r_ind_k
        index_levels[k + 1] = index_levels[k] * (1 + r_ind_k)
        
        # Step B: Update EWMA estimates
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
        
        # Leverage control MAs (using R_k^ind level)
        R_k_ind = index_levels[k + 1]
        
        raw_long = beta_long * raw_long + (1 - beta_long) * R_k_ind
        correction_long = 1 - beta_long**(k + 1)
        ma_long_k = raw_long / correction_long
        ma_long[k] = ma_long_k
        
        raw_short = beta_short * raw_short + (1 - beta_short) * R_k_ind
        correction_short = 1 - beta_short**(k + 1)
        ma_short_k = raw_short / correction_short
        ma_short[k] = ma_short_k
        
        # Step C: Compute tracking error
        e_k = np.log(sigma_hat_k_ind / sigma_tar)
        tracking_error[k] = e_k
        
        # Step D: Update controllers and set weights
        if k < warm_start_days:
            # Warm start: use open-loop weights
            kappa_k = 0.0
            kappa_k_lev = 0.0
            w_k = min(sigma_tar / sigma_hat_k, max_leverage)
        else:
            # Volatility control state
            clipped_input = np.clip(-gain * e_k, kappa_min, kappa_max)
            kappa_k = (1 - theta) * clipped_input + theta * kappa_prev
            
            # Leverage control state
            log_ratio = np.log(ma_long_k / ma_short_k)
            kappa_k_lev = np.clip(-gain_lev * log_ratio, kappa_min_lev, 0.0)
            
            # Combined weight
            w_vol = np.exp(kappa_k) * sigma_tar / sigma_hat_k
            w_lev = np.exp(kappa_k_lev) * max_leverage
            w_k = min(w_vol, w_lev)
        
        c_k = 1 - w_k
        
        weights[k] = w_k
        cash_weights[k] = c_k
        kappa[k] = kappa_k
        kappa_lev[k] = kappa_k_lev
        
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
        'kappa': kappa,
        'kappa_lev': kappa_lev,
        'ma_long': ma_long,
        'ma_short': ma_short
    }
    
    return results


def validate_leverage_implementation() -> bool:
    """
    Validate leverage control algorithm implementation.
    
    Returns
    -------
    bool
        True if all validations pass
    """
    # Test 1: Warm start period
    r = np.ones(20) * 0.01
    rf = np.ones(20) * 0.0001
    dates = pd.date_range('2020-01-01', periods=20)
    
    results = run_leverage_control_strategy(r, rf, dates, warm_start_days=10)
    
    # During warm start, both kappa should be 0
    assert np.all(results['kappa'][:10] == 0.0), \
        "Kappa should be 0 during warm start"
    assert np.all(results['kappa_lev'][:10] == 0.0), \
        "Kappa_lev should be 0 during warm start"
    
    # Test 2: Verify weights are bounded
    assert np.all(results['weights'] >= 0), "Weights should be non-negative"
    assert np.all(results['weights'] <= 1.5), "Weights should not exceed max leverage"
    
    # Test 3: Verify kappa_lev bounds
    assert np.all(results['kappa_lev'] >= -2.0), "Kappa_lev should be >= -2"
    assert np.all(results['kappa_lev'] <= 0.0), "Kappa_lev should be <= 0"
    
    # Test 4: Verify weights respect leverage limit
    for k in range(10, len(results['weights'])):
        w_lev = np.exp(results['kappa_lev'][k]) * 1.5
        assert results['weights'][k] <= w_lev + 1e-10, \
            f"Weight at k={k} exceeds leverage limit"
    
    # Test 5: Verify cash weights
    assert np.allclose(results['weights'] + results['cash_weights'], 1.0), \
        "Weights and cash weights should sum to 1"
    
    # Test 6: Verify R_0 = 1
    assert results['index_levels'][0] == 1.0, "Initial index level should be 1"
    
    print("✓ Leverage control algorithm validation passed")
    return True

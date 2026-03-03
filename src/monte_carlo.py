"""
Monte Carlo EWMA Uncertainty Quantification
==========================================

Implements Appendix A Monte Carlo benchmarking for EWMA estimator.
"""

import numpy as np
from typing import Tuple


def run_monte_carlo_ewma(
    n_simulations: int = 10000,
    n_burnin: int = 252,
    n_backtest: int = 6270,
    halflife: int = 126,
    target_vol_annual: float = 0.15,
    trading_days: int = 252,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulation to quantify EWMA estimator uncertainty.
    
    Generates N synthetic return series from N(0, sigma_tar^2) and computes
    EWMA volatility with finite-sample correction for each. Returns percentiles
    to construct confidence bands.
    
    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo paths (default: 10000)
    n_burnin : int
        Burn-in period to discard (default: 252)
    n_backtest : int
        Number of backtest periods (default: 6270)
    halflife : int
        EWMA halflife (default: 126)
    target_vol_annual : float
        Target annualized volatility (default: 0.15)
    trading_days : int
        Trading days per year (default: 252)
    random_seed : int
        Random seed for reproducibility (default: 42)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (p10, p50, p90) - 10th, 50th, and 90th percentiles of annualized
        EWMA volatility across simulations, each of length n_backtest
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Daily target volatility
    sigma_tar = target_vol_annual / np.sqrt(trading_days)
    
    # Total time steps
    T_total = n_burnin + n_backtest
    
    # Compute beta
    beta = np.exp(-np.log(2) / halflife)
    
    print(f"Running Monte Carlo simulation:")
    print(f"  N simulations: {n_simulations}")
    print(f"  T total: {T_total} (burnin: {n_burnin}, backtest: {n_backtest})")
    print(f"  Halflife: {halflife}")
    print(f"  Target vol (daily): {sigma_tar:.6f}")
    print(f"  Target vol (annual): {target_vol_annual:.4f}")
    print(f"  Random seed: {random_seed}")
    
    # Generate synthetic returns: shape (n_simulations, T_total)
    R_synth = np.random.normal(0, sigma_tar, size=(n_simulations, T_total))
    
    # Compute EWMA volatility for all paths simultaneously (vectorized)
    raw = np.zeros(n_simulations)
    sigma_hat_all = np.zeros((n_simulations, T_total))
    
    for k in range(T_total):
        # Update raw EWMA for all paths
        raw = beta * raw + (1 - beta) * R_synth[:, k]**2
        
        # Apply finite-sample correction
        correction = 1 - beta**(k + 1)
        sigma_hat_all[:, k] = np.sqrt(raw / correction) * np.sqrt(trading_days)
    
    # Discard burn-in period
    sigma_post = sigma_hat_all[:, n_burnin:]  # shape: (n_simulations, n_backtest)
    
    # Compute percentiles across paths at each time step
    p10 = np.percentile(sigma_post, 10, axis=0)
    p50 = np.percentile(sigma_post, 50, axis=0)
    p90 = np.percentile(sigma_post, 90, axis=0)
    
    # Sanity checks
    mean_vol = np.mean(sigma_post)
    median_vol = np.median(sigma_post)
    
    print(f"\nMonte Carlo results:")
    print(f"  Mean annualized vol: {mean_vol:.4f}")
    print(f"  Median annualized vol: {median_vol:.4f}")
    print(f"  10th percentile (avg): {np.mean(p10):.4f}")
    print(f"  90th percentile (avg): {np.mean(p90):.4f}")
    print(f"  Band width (avg): {np.mean(p90 - p10):.4f}")
    
    return p10, p50, p90


def validate_monte_carlo() -> bool:
    """
    Validate Monte Carlo implementation with sanity checks.
    
    Returns
    -------
    bool
        True if all validations pass
    """
    # Run small simulation
    p10, p50, p90 = run_monte_carlo_ewma(
        n_simulations=1000,
        n_burnin=50,
        n_backtest=100,
        random_seed=42
    )
    
    # Test 1: Check output shapes
    assert len(p10) == 100, "p10 length should be n_backtest"
    assert len(p50) == 100, "p50 length should be n_backtest"
    assert len(p90) == 100, "p90 length should be n_backtest"
    
    # Test 2: Check ordering
    assert np.all(p10 <= p50), "p10 should be <= p50"
    assert np.all(p50 <= p90), "p50 should be <= p90"
    
    # Test 3: Check values are reasonable (near 15%)
    assert np.all(p50 > 0.10), "p50 should be > 10%"
    assert np.all(p50 < 0.20), "p50 should be < 20%"
    
    # Test 4: Check approximate symmetry around target
    mean_p50 = np.mean(p50)
    assert abs(mean_p50 - 0.15) < 0.02, \
        f"Mean p50 should be near 0.15, got {mean_p50}"
    
    print("✓ Monte Carlo validation passed")
    return True

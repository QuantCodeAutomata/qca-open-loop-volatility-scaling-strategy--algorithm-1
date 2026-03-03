"""
EWMA Volatility Estimator
=========================

Implements the finite-sample-normalized EWMA variance estimator from Equation 1.
"""

import numpy as np
from typing import Tuple


def compute_ewma_vol(returns: np.ndarray, halflife: int = 126) -> np.ndarray:
    """
    Compute running EWMA volatility with finite-sample correction (Equation 1).
    
    At each day k (k = 1, 2, ..., N):
      sigma_hat_k^2 = [(1-beta)/(1-beta^k)] * sum_{j=1}^{k} beta^{k-j} * r_j^2
    
    Implemented recursively:
      raw_k = beta * raw_{k-1} + (1-beta) * r_k^2  [with raw_0 = 0]
      sigma_hat_k^2 = raw_k / (1 - beta^k)  [finite-sample correction]
      sigma_hat_k = sqrt(sigma_hat_k^2)
    
    Parameters
    ----------
    returns : np.ndarray
        Array of simple returns (length N)
    halflife : int
        EWMA halflife in days (default: 126)
    
    Returns
    -------
    np.ndarray
        Array of EWMA volatility estimates (length N)
    """
    N = len(returns)
    beta = np.exp(-np.log(2) / halflife)
    
    sigma_hat = np.zeros(N)
    raw = 0.0
    
    for k in range(N):
        # Update raw EWMA
        raw = beta * raw + (1 - beta) * returns[k]**2
        
        # Apply finite-sample correction
        correction = 1 - beta**(k + 1)  # k+1 because paper uses 1-indexing
        sigma_hat[k] = np.sqrt(raw / correction)
    
    return sigma_hat


def compute_ewma_level(levels: np.ndarray, halflife: int) -> np.ndarray:
    """
    Compute running EWMA of a level series (e.g., cumulative index level R_k^ind).
    
    Used for leverage control moving averages in Algorithm 3.
    
    MA_k = [(1-beta)/(1-beta^k)] * sum_{j=1}^{k} beta^{k-j} * level_j
    
    Implemented recursively:
      raw_k = beta * raw_{k-1} + (1-beta) * level_k
      MA_k = raw_k / (1 - beta^k)
    
    Parameters
    ----------
    levels : np.ndarray
        Array of level values (e.g., R_k^ind)
    halflife : int
        EWMA halflife in days
    
    Returns
    -------
    np.ndarray
        Array of EWMA level estimates
    """
    N = len(levels)
    beta = np.exp(-np.log(2) / halflife)
    
    ma = np.zeros(N)
    raw = 0.0
    
    for k in range(N):
        # Update raw EWMA
        raw = beta * raw + (1 - beta) * levels[k]
        
        # Apply finite-sample correction
        correction = 1 - beta**(k + 1)
        ma[k] = raw / correction
    
    return ma


def validate_ewma_implementation(halflife: int = 126) -> bool:
    """
    Validate EWMA implementation with known test cases.
    
    Parameters
    ----------
    halflife : int
        EWMA halflife to test
    
    Returns
    -------
    bool
        True if all validations pass
    """
    # Test 1: Single return should give sigma_hat_1 = |r_1|
    r = np.array([0.02])
    sigma = compute_ewma_vol(r, halflife)
    assert np.isclose(sigma[0], abs(r[0])), f"Expected {abs(r[0])}, got {sigma[0]}"
    
    # Test 2: Beta halflife check
    beta = np.exp(-np.log(2) / halflife)
    halflife_check = -np.log(2) / np.log(beta)
    assert np.isclose(halflife_check, halflife), f"Halflife mismatch: {halflife_check} vs {halflife}"
    
    # Test 3: Constant returns should converge to std
    r_const = np.ones(1000) * 0.01
    sigma = compute_ewma_vol(r_const, halflife)
    # Should converge to 0.01
    assert np.isclose(sigma[-1], 0.01, rtol=0.01), f"Expected ~0.01, got {sigma[-1]}"
    
    print(f"✓ EWMA validation passed for halflife={halflife}")
    return True


def compute_annualized_vol(daily_vol: np.ndarray, trading_days: int = 252) -> np.ndarray:
    """
    Convert daily volatility to annualized volatility.
    
    Parameters
    ----------
    daily_vol : np.ndarray
        Daily volatility estimates
    trading_days : int
        Number of trading days per year (default: 252)
    
    Returns
    -------
    np.ndarray
        Annualized volatility
    """
    return daily_vol * np.sqrt(trading_days)

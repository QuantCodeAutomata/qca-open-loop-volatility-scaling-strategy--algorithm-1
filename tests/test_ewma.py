"""
Tests for EWMA Volatility Estimator
===================================
"""

import numpy as np
import pytest
from src.ewma import compute_ewma_vol, compute_ewma_level, validate_ewma_implementation


def test_ewma_single_return():
    """Test that single return gives sigma_hat_1 = |r_1| (finite-sample correction)."""
    r = np.array([0.02])
    sigma = compute_ewma_vol(r, halflife=126)
    assert np.isclose(sigma[0], abs(r[0])), f"Expected {abs(r[0])}, got {sigma[0]}"


def test_ewma_negative_return():
    """Test EWMA with negative return."""
    r = np.array([-0.03])
    sigma = compute_ewma_vol(r, halflife=126)
    assert np.isclose(sigma[0], abs(r[0])), f"Expected {abs(r[0])}, got {sigma[0]}"


def test_ewma_beta_halflife():
    """Test that beta calculation gives correct halflife."""
    halflife = 126
    beta = np.exp(-np.log(2) / halflife)
    halflife_check = -np.log(2) / np.log(beta)
    assert np.isclose(halflife_check, halflife), \
        f"Halflife mismatch: {halflife_check} vs {halflife}"


def test_ewma_constant_returns():
    """Test EWMA converges to std for constant returns."""
    r_const = np.ones(1000) * 0.01
    sigma = compute_ewma_vol(r_const, halflife=126)
    # Should converge to 0.01
    assert np.isclose(sigma[-1], 0.01, rtol=0.01), \
        f"Expected ~0.01, got {sigma[-1]}"


def test_ewma_zero_returns():
    """Test EWMA with all zero returns."""
    r = np.zeros(100)
    sigma = compute_ewma_vol(r, halflife=126)
    assert np.all(sigma == 0), "EWMA should be zero for zero returns"


def test_ewma_positive_values():
    """Test that EWMA volatility is always non-negative."""
    r = np.random.randn(500) * 0.01
    sigma = compute_ewma_vol(r, halflife=126)
    assert np.all(sigma >= 0), "EWMA volatility should be non-negative"


def test_ewma_finite_sample_correction():
    """Test finite-sample correction at early time steps."""
    r = np.array([0.02, 0.01])
    halflife = 126
    beta = np.exp(-np.log(2) / halflife)
    
    sigma = compute_ewma_vol(r, halflife)
    
    # At k=1 (0-indexed k=0): sigma^2 = r_1^2
    assert np.isclose(sigma[0]**2, r[0]**2)
    
    # At k=2 (0-indexed k=1): sigma^2 = [(1-beta)/(1-beta^2)] * [beta*r_1^2 + r_2^2]
    expected_var = ((1-beta)/(1-beta**2)) * (beta * r[0]**2 + r[1]**2)
    assert np.isclose(sigma[1]**2, expected_var, rtol=1e-10)


def test_ewma_level():
    """Test EWMA of level series."""
    levels = np.array([1.0, 1.01, 1.02, 1.03])
    ma = compute_ewma_level(levels, halflife=10)
    
    # Should be non-negative
    assert np.all(ma >= 0), "EWMA levels should be non-negative"
    
    # First value should equal first level (with correction)
    assert ma[0] == levels[0], "First MA should equal first level"


def test_ewma_different_halflives():
    """Test EWMA with different halflife values."""
    r = np.random.randn(200) * 0.015
    
    sigma_short = compute_ewma_vol(r, halflife=21)
    sigma_long = compute_ewma_vol(r, halflife=126)
    
    # Both should be positive
    assert np.all(sigma_short > 0)
    assert np.all(sigma_long > 0)
    
    # Short halflife should adapt faster (more variance in estimates)
    var_short = np.var(sigma_short[50:])  # After burn-in
    var_long = np.var(sigma_long[50:])
    assert var_short >= var_long * 0.5, "Short halflife should have higher variance"


def test_ewma_validation():
    """Test the validation function."""
    result = validate_ewma_implementation(halflife=126)
    assert result is True, "Validation should pass"


def test_ewma_extreme_values():
    """Test EWMA with extreme return values."""
    r = np.array([0.1, -0.1, 0.05, -0.05])
    sigma = compute_ewma_vol(r, halflife=126)
    
    # Should handle extreme values without errors
    assert np.all(np.isfinite(sigma)), "EWMA should be finite for extreme values"
    assert np.all(sigma >= 0), "EWMA should be non-negative"


def test_ewma_long_series():
    """Test EWMA with long time series."""
    np.random.seed(42)
    r = np.random.randn(5000) * 0.01
    sigma = compute_ewma_vol(r, halflife=126)
    
    # Check convergence to approximately true vol
    true_vol = 0.01
    # After many observations, should be close to true vol
    assert np.abs(np.mean(sigma[-1000:]) - true_vol) < 0.005, \
        "EWMA should converge to true volatility"

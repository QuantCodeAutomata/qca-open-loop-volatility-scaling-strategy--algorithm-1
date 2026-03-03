"""
Tests for Performance Metrics
=============================
"""

import numpy as np
import pytest
from src.metrics import (
    compute_volatility_tracking_error,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_maximum_drawdown,
    compute_all_metrics
)


def test_tracking_error_zero():
    """Test tracking error when volatility matches target exactly."""
    # Create returns with exactly 15% annualized vol
    np.random.seed(42)
    target_vol = 0.15
    daily_vol = target_vol / np.sqrt(252)
    returns = np.random.normal(0, daily_vol, 1000)
    
    tracking_error = compute_volatility_tracking_error(returns, target_vol_annual=target_vol)
    
    # Should be small (not exactly zero due to EWMA estimation)
    assert tracking_error < 0.05, f"Expected small tracking error, got {tracking_error}"


def test_tracking_error_positive():
    """Test that tracking error is always non-negative."""
    np.random.seed(42)
    returns = np.random.randn(500) * 0.015
    
    tracking_error = compute_volatility_tracking_error(returns)
    
    assert tracking_error >= 0, "Tracking error should be non-negative"


def test_annualized_return_positive():
    """Test annualized return calculation with positive returns."""
    # Create index levels with 10% total return over 252 days
    levels = np.linspace(1.0, 1.1, 253)  # 253 levels = 252 returns
    
    ann_return = compute_annualized_return(levels, trading_days=252)
    
    # Should be approximately 10%
    assert np.isclose(ann_return, 0.10, rtol=0.01)


def test_annualized_return_negative():
    """Test annualized return with negative returns."""
    # 20% loss over 252 days
    levels = np.linspace(1.0, 0.8, 253)
    
    ann_return = compute_annualized_return(levels, trading_days=252)
    
    # Should be approximately -20%
    assert np.isclose(ann_return, -0.20, rtol=0.01)


def test_annualized_return_flat():
    """Test annualized return with flat returns."""
    levels = np.ones(253)
    
    ann_return = compute_annualized_return(levels, trading_days=252)
    
    # Should be approximately 0%
    assert np.isclose(ann_return, 0.0, atol=0.001)


def test_annualized_volatility():
    """Test annualized volatility calculation."""
    # Create returns with known std
    np.random.seed(42)
    daily_std = 0.01
    returns = np.random.normal(0, daily_std, 1000)
    
    ann_vol = compute_annualized_volatility(returns, trading_days=252)
    
    # Should be approximately daily_std * sqrt(252)
    expected = daily_std * np.sqrt(252)
    assert np.isclose(ann_vol, expected, rtol=0.1)  # Allow 10% tolerance due to randomness


def test_annualized_volatility_zero():
    """Test annualized volatility with zero returns."""
    returns = np.zeros(100)
    
    ann_vol = compute_annualized_volatility(returns)
    
    assert ann_vol == 0.0, "Volatility should be zero for zero returns"


def test_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    ann_return = 0.10
    ann_vol = 0.15
    
    sharpe = compute_sharpe_ratio(ann_return, ann_vol)
    
    expected = 0.10 / 0.15
    assert np.isclose(sharpe, expected)


def test_sharpe_ratio_zero_vol():
    """Test Sharpe ratio with zero volatility."""
    sharpe = compute_sharpe_ratio(0.10, 0.0)
    
    assert sharpe == 0.0, "Sharpe should be 0 when vol is 0"


def test_sharpe_ratio_negative_return():
    """Test Sharpe ratio with negative return."""
    sharpe = compute_sharpe_ratio(-0.05, 0.15)
    
    expected = -0.05 / 0.15
    assert np.isclose(sharpe, expected)


def test_maximum_drawdown_no_drawdown():
    """Test maximum drawdown with monotonically increasing levels."""
    levels = np.linspace(1.0, 2.0, 100)
    
    max_dd = compute_maximum_drawdown(levels)
    
    assert max_dd == 0.0, "No drawdown for monotonically increasing series"


def test_maximum_drawdown_single_drop():
    """Test maximum drawdown with single drop."""
    levels = np.array([1.0, 1.1, 1.2, 0.9, 1.0, 1.1])
    
    max_dd = compute_maximum_drawdown(levels)
    
    # Peak is 1.2, trough is 0.9, DD = (1.2 - 0.9) / 1.2 = 0.25
    expected = (1.2 - 0.9) / 1.2
    assert np.isclose(max_dd, expected)


def test_maximum_drawdown_recovery():
    """Test maximum drawdown after recovery."""
    levels = np.array([1.0, 2.0, 1.0, 2.5])
    
    max_dd = compute_maximum_drawdown(levels)
    
    # Max DD is from peak 2.0 to trough 1.0 = 50%
    expected = (2.0 - 1.0) / 2.0
    assert np.isclose(max_dd, expected)


def test_maximum_drawdown_positive():
    """Test that maximum drawdown is always non-negative."""
    np.random.seed(42)
    returns = np.random.randn(200) * 0.01
    levels = np.cumprod(1 + returns)
    levels = np.insert(levels, 0, 1.0)  # Add R_0 = 1
    
    max_dd = compute_maximum_drawdown(levels)
    
    assert max_dd >= 0, "Maximum drawdown should be non-negative"
    assert max_dd <= 1, "Maximum drawdown should be <= 1"


def test_all_metrics():
    """Test compute_all_metrics function."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.01 + 0.0003
    levels = np.cumprod(1 + returns)
    levels = np.insert(levels, 0, 1.0)
    
    metrics = compute_all_metrics(returns, levels)
    
    # Check all keys are present
    assert 'tracking_error' in metrics
    assert 'ann_return' in metrics
    assert 'ann_vol' in metrics
    assert 'sharpe' in metrics
    assert 'max_dd' in metrics
    
    # Check all values are reasonable
    assert metrics['tracking_error'] >= 0
    assert -1 < metrics['ann_return'] < 2  # Reasonable range
    assert 0 < metrics['ann_vol'] < 1  # Reasonable range
    assert -5 < metrics['sharpe'] < 5  # Reasonable range
    assert 0 <= metrics['max_dd'] <= 1


def test_metrics_consistency():
    """Test that metrics are consistent with manual calculation."""
    # Simple test case
    returns = np.array([0.01, -0.005, 0.015, -0.01, 0.02])
    levels = np.cumprod(1 + returns)
    levels = np.insert(levels, 0, 1.0)
    
    metrics = compute_all_metrics(returns, levels)
    
    # Manual calculation of annualized return
    N = len(returns)
    R_N = levels[-1]
    expected_ann_return = R_N ** (252 / N) - 1
    
    assert np.isclose(metrics['ann_return'], expected_ann_return)
    
    # Manual calculation of annualized vol
    expected_ann_vol = np.std(returns, ddof=1) * np.sqrt(252)
    
    assert np.isclose(metrics['ann_vol'], expected_ann_vol)


def test_tracking_error_edge_cases():
    """Test tracking error with edge cases."""
    # Single return
    returns = np.array([0.01])
    te = compute_volatility_tracking_error(returns)
    assert te >= 0
    
    # Very high volatility
    returns = np.random.randn(100) * 0.1
    te = compute_volatility_tracking_error(returns)
    assert te >= 0
    
    # Very low volatility
    returns = np.random.randn(100) * 0.001
    te = compute_volatility_tracking_error(returns)
    assert te >= 0

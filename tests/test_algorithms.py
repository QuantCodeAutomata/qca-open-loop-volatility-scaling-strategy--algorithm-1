"""
Tests for Algorithm Implementations
===================================
"""

import numpy as np
import pandas as pd
import pytest
from src.algorithm1_openloop import run_openloop_strategy, run_buyhold_baseline, validate_openloop_implementation
from src.algorithm2_feedback import run_feedback_control_strategy, validate_feedback_implementation
from src.algorithm3_leverage import run_leverage_control_strategy, validate_leverage_implementation


# Test data fixture
@pytest.fixture
def test_data():
    """Create test data for algorithms."""
    np.random.seed(42)
    n = 100
    r = np.random.randn(n) * 0.01 + 0.0003  # Mean return ~3% annual
    rf = np.ones(n) * 0.0001  # ~2.5% annual risk-free
    dates = pd.date_range('2020-01-01', periods=n)
    return r, rf, dates


def test_openloop_basic(test_data):
    """Test basic open-loop algorithm execution."""
    r, rf, dates = test_data
    results = run_openloop_strategy(r, rf, dates)
    
    # Check output structure
    assert 'weights' in results
    assert 'index_returns' in results
    assert 'index_levels' in results
    assert len(results['weights']) == len(r)
    assert len(results['index_levels']) == len(r) + 1


def test_openloop_weights_bounded(test_data):
    """Test that open-loop weights are properly bounded."""
    r, rf, dates = test_data
    results = run_openloop_strategy(r, rf, dates, max_leverage=1.5)
    
    assert np.all(results['weights'] >= 0), "Weights should be non-negative"
    assert np.all(results['weights'] <= 1.5), "Weights should not exceed max leverage"


def test_openloop_cash_weights(test_data):
    """Test that weights and cash weights sum to 1."""
    r, rf, dates = test_data
    results = run_openloop_strategy(r, rf, dates)
    
    assert np.allclose(results['weights'] + results['cash_weights'], 1.0), \
        "Weights and cash weights should sum to 1"


def test_openloop_initial_conditions(test_data):
    """Test initial conditions for open-loop."""
    r, rf, dates = test_data
    results = run_openloop_strategy(r, rf, dates)
    
    # R_0 should be 1
    assert results['index_levels'][0] == 1.0, "Initial index level should be 1"


def test_openloop_timing(test_data):
    """Test that returns use previous period's weights (no look-ahead)."""
    r = np.array([0.01, 0.02, -0.01])
    rf = np.array([0.0001, 0.0001, 0.0001])
    dates = pd.date_range('2020-01-01', periods=3)
    
    results = run_openloop_strategy(r, rf, dates)
    
    # First return should use w_0 = 1.0
    expected_r0 = r[0] * 1.0 + rf[0] * 0.0
    assert np.isclose(results['index_returns'][0], expected_r0)


def test_openloop_validation():
    """Test open-loop validation function."""
    result = validate_openloop_implementation()
    assert result is True


def test_buyhold_baseline(test_data):
    """Test buy-and-hold baseline."""
    r, rf, dates = test_data
    results = run_buyhold_baseline(r, rf, dates)
    
    # Weights should all be 1
    assert np.all(results['weights'] == 1.0), "Buy-hold weights should be 1"
    
    # Index returns should equal risky returns
    assert np.allclose(results['index_returns'], r), \
        "Buy-hold returns should equal risky returns"


def test_feedback_basic(test_data):
    """Test basic feedback control algorithm execution."""
    r, rf, dates = test_data
    results = run_feedback_control_strategy(r, rf, dates)
    
    # Check output structure
    assert 'kappa' in results
    assert 'tracking_error' in results
    assert len(results['kappa']) == len(r)


def test_feedback_warm_start(test_data):
    """Test feedback warm start period."""
    r, rf, dates = test_data
    results = run_feedback_control_strategy(r, rf, dates, warm_start_days=10)
    
    # During warm start, kappa should be 0
    assert np.all(results['kappa'][:10] == 0.0), \
        "Kappa should be 0 during warm start"


def test_feedback_weights_bounded(test_data):
    """Test that feedback weights are properly bounded."""
    r, rf, dates = test_data
    results = run_feedback_control_strategy(r, rf, dates)
    
    assert np.all(results['weights'] >= 0), "Weights should be non-negative"
    assert np.all(results['weights'] <= 1.5), "Weights should not exceed max leverage"


def test_feedback_kappa_bounds(test_data):
    """Test that kappa stays approximately bounded."""
    r, rf, dates = test_data
    results = run_feedback_control_strategy(r, rf, dates, kappa_min=-1, kappa_max=1)
    
    # After warm start, kappa should be roughly bounded
    # (not exactly due to smoothing, but close)
    assert np.all(results['kappa'] >= -2), "Kappa should be roughly >= -2"
    assert np.all(results['kappa'] <= 2), "Kappa should be roughly <= 2"


def test_feedback_tracking_error():
    """Test tracking error calculation."""
    r = np.ones(50) * 0.01
    rf = np.ones(50) * 0.0001
    dates = pd.date_range('2020-01-01', periods=50)
    
    results = run_feedback_control_strategy(r, rf, dates)
    
    # Tracking error should be log ratio
    sigma_tar = 0.15 / np.sqrt(252)
    for k in range(10, 50):  # After warm start
        expected_e = np.log(results['index_vol'][k] / sigma_tar)
        assert np.isclose(results['tracking_error'][k], expected_e, rtol=1e-6)


def test_feedback_validation():
    """Test feedback validation function."""
    result = validate_feedback_implementation()
    assert result is True


def test_leverage_basic(test_data):
    """Test basic leverage control algorithm execution."""
    r, rf, dates = test_data
    results = run_leverage_control_strategy(r, rf, dates)
    
    # Check output structure
    assert 'kappa_lev' in results
    assert 'ma_long' in results
    assert 'ma_short' in results
    assert len(results['kappa_lev']) == len(r)


def test_leverage_warm_start(test_data):
    """Test leverage control warm start."""
    r, rf, dates = test_data
    results = run_leverage_control_strategy(r, rf, dates, warm_start_days=10)
    
    # During warm start, both kappa should be 0
    assert np.all(results['kappa'][:10] == 0.0), \
        "Kappa should be 0 during warm start"
    assert np.all(results['kappa_lev'][:10] == 0.0), \
        "Kappa_lev should be 0 during warm start"


def test_leverage_kappa_lev_bounds(test_data):
    """Test that kappa_lev is bounded in [-2, 0]."""
    r, rf, dates = test_data
    results = run_leverage_control_strategy(r, rf, dates, kappa_min_lev=-2)
    
    assert np.all(results['kappa_lev'] >= -2.0), "Kappa_lev should be >= -2"
    assert np.all(results['kappa_lev'] <= 0.0), "Kappa_lev should be <= 0"


def test_leverage_weights_respect_limit(test_data):
    """Test that weights respect leverage limit."""
    r, rf, dates = test_data
    results = run_leverage_control_strategy(r, rf, dates)
    
    # After warm start, weights should respect exp(kappa_lev) * L
    for k in range(10, len(results['weights'])):
        w_lev = np.exp(results['kappa_lev'][k]) * 1.5
        assert results['weights'][k] <= w_lev + 1e-9, \
            f"Weight at k={k} exceeds leverage limit"


def test_leverage_moving_averages(test_data):
    """Test that moving averages are computed correctly."""
    r, rf, dates = test_data
    results = run_leverage_control_strategy(r, rf, dates)
    
    # MAs should be positive
    assert np.all(results['ma_long'] > 0), "Long MA should be positive"
    assert np.all(results['ma_short'] > 0), "Short MA should be positive"


def test_leverage_validation():
    """Test leverage validation function."""
    result = validate_leverage_implementation()
    assert result is True


def test_all_algorithms_same_data():
    """Test all algorithms on same data and compare."""
    np.random.seed(42)
    n = 200
    r = np.random.randn(n) * 0.015 + 0.0003
    rf = np.ones(n) * 0.0001
    dates = pd.date_range('2020-01-01', periods=n)
    
    results_buyhold = run_buyhold_baseline(r, rf, dates)
    results_openloop = run_openloop_strategy(r, rf, dates)
    results_feedback = run_feedback_control_strategy(r, rf, dates)
    results_leverage = run_leverage_control_strategy(r, rf, dates)
    
    # All should have same length
    assert len(results_buyhold['index_returns']) == n
    assert len(results_openloop['index_returns']) == n
    assert len(results_feedback['index_returns']) == n
    assert len(results_leverage['index_returns']) == n
    
    # All should start at R_0 = 1
    assert results_buyhold['index_levels'][0] == 1.0
    assert results_openloop['index_levels'][0] == 1.0
    assert results_feedback['index_levels'][0] == 1.0
    assert results_leverage['index_levels'][0] == 1.0


def test_algorithm_consistency():
    """Test that algorithms produce consistent results with fixed seed."""
    np.random.seed(123)
    r1 = np.random.randn(50) * 0.01
    rf1 = np.ones(50) * 0.0001
    dates1 = pd.date_range('2020-01-01', periods=50)
    
    results1 = run_openloop_strategy(r1, rf1, dates1)
    
    # Run again with same seed
    np.random.seed(123)
    r2 = np.random.randn(50) * 0.01
    rf2 = np.ones(50) * 0.0001
    dates2 = pd.date_range('2020-01-01', periods=50)
    
    results2 = run_openloop_strategy(r2, rf2, dates2)
    
    # Should be identical
    assert np.allclose(results1['weights'], results2['weights'])
    assert np.allclose(results1['index_returns'], results2['index_returns'])

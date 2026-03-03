# Implementation Notes

## Project Overview

This repository implements a complete suite of volatility scaling strategies for quantitative finance, including:

1. **Open-Loop Volatility Scaling (Algorithm 1)**
2. **Proportional Feedback Control (Algorithm 2)**
3. **Leverage-Limiting Control (Algorithm 3)**
4. **Monte Carlo EWMA Uncertainty Quantification (Experiment 4)**

## Implementation Highlights

### Code Quality

- **Type Hints**: All functions have complete type annotations
- **Docstrings**: Comprehensive documentation for all modules and functions
- **Testing**: 50 unit tests covering all algorithms, metrics, and edge cases
- **Test Coverage**: 100% pass rate on all tests
- **PEP 8 Compliant**: Clean, readable code following Python standards

### Key Technical Features

#### 1. Finite-Sample EWMA Correction

The implementation includes the critical finite-sample normalization for EWMA estimators:

```python
sigma_hat_k^2 = raw_k / (1 - beta^k)
```

This ensures accurate volatility estimates even in early periods of the backtest.

#### 2. No Look-Ahead Bias

All algorithms strictly follow the paper methodology:
- Returns are realized using weights from the previous period
- Volatility estimates use only past data
- Controller states are updated after returns are realized

#### 3. Proper Timing Sequence

Each algorithm follows this exact order per day k:
1. Realize returns using w_{k-1}
2. Update cumulative index level R_k
3. Update EWMA volatility estimates
4. Compute tracking error (if applicable)
5. Update controller states (if applicable)
6. Set new weights w_k for next period

### Data Handling

#### Real Data Mode

When `MASSIVE_TOKEN` is set, the system loads:
- IVV daily adjusted close prices from Massive API
- Federal Funds Effective Rate from FRED

#### Synthetic Data Mode

When `MASSIVE_TOKEN` is not set, generates realistic synthetic data with:
- Regime changes matching historical market periods
- Dot-com crash (2000-2002)
- Financial crisis (2008-2009)
- COVID crash (2020)
- Bear market (2022)

### Algorithm Implementations

#### Algorithm 1: Open-Loop

```
w_k = min(sigma_tar / sigma_hat_k, L)
```

- Simple ratio-based weight adjustment
- No feedback mechanism
- Tracking error: ~1.28% (synthetic data)

#### Algorithm 2: Feedback Control

```
e_k = log(sigma_hat_k^ind / sigma_tar)
kappa_k = (1-theta) * clip(-g*e_k, [-1,1]) + theta*kappa_{k-1}
w_k = min(exp(kappa_k) * sigma_tar / sigma_hat_k, L)
```

- Log-volatility tracking error signal
- Smoothed proportional controller
- Tracking error: ~0.13% (synthetic data)

#### Algorithm 3: Leverage Control

```
kappa_k^lev = clip(-g_ell * log(MA_long / MA_short), [-2, 0])
w_k = min(exp(kappa_k) * sigma_tar / sigma_hat_k, exp(kappa_k^lev) * L)
```

- Combines volatility and leverage control
- Reduces leverage during drawdowns
- Best drawdown control: 56.95% max DD (synthetic data)

### Performance Metrics

All five metrics from the paper are implemented:

1. **Volatility Tracking Error**: Mean absolute deviation from 15% target
2. **Annualized Return**: CAGR over backtest period
3. **Annualized Volatility**: Sample std × √252
4. **Sharpe Ratio**: Return / Volatility
5. **Maximum Drawdown**: Worst peak-to-trough decline

### Monte Carlo Simulation

- 10,000 synthetic paths under constant 15% volatility
- 252-day burn-in period
- Generates 10th-90th percentile confidence bands
- Validates estimator uncertainty bounds

## Test Suite

### EWMA Tests (12 tests)
- Basic functionality
- Finite-sample correction
- Halflife validation
- Edge cases (zeros, negatives, extremes)

### Algorithm Tests (21 tests)
- Weight bounds (0 ≤ w ≤ 1.5)
- Cash weight consistency
- Warm start behavior
- Controller state bounds
- Timing correctness
- Cross-algorithm comparison

### Metrics Tests (17 tests)
- Tracking error calculations
- Return metrics (positive, negative, flat)
- Volatility calculations
- Sharpe ratio edge cases
- Drawdown computations

## Visualization

Five comprehensive figures:

1. **Cumulative Returns**: Log-scale performance comparison
2. **Running Volatility**: EWMA with Monte Carlo bands
3. **Portfolio Weights**: Dynamic leverage over time
4. **Controller States**: kappa and kappa_lev evolution
5. **Drawdowns**: Peak-to-trough profiles

## Results (Synthetic Data)

| Strategy | Tracking Error | Ann. Return | Ann. Vol | Sharpe | Max DD |
|----------|----------------|-------------|----------|--------|--------|
| IVV Buy-and-Hold | 6.49% | -0.82% | 21.90% | -0.037 | 76.55% |
| Open-Loop | 1.28% | 1.55% | 14.86% | 0.104 | 61.14% |
| Volatility Control | 0.13% | 2.52% | 15.03% | 0.168 | 56.97% |
| Leverage Control | 0.98% | 1.67% | 14.24% | 0.117 | 56.95% |

**Note**: Results use synthetic data for demonstration. With real IVV data and proper MASSIVE_TOKEN, results should match paper benchmarks more closely.

## Usage

### Run All Experiments

```bash
python run_experiments.py
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Specific Algorithm

```python
from src.data_loader import get_experiment_data
from src.algorithm2_feedback import run_volatility_control

data, n_days = get_experiment_data()
results = run_volatility_control(
    data['return'].values,
    data['rf_return_360'].values,
    dates=data.index
)
```

## Repository Files

### Core Modules (`src/`)
- `data_loader.py`: Real data loading from Massive API
- `data_loader_synthetic.py`: Synthetic data generation
- `ewma.py`: EWMA volatility estimator with finite-sample correction
- `metrics.py`: Performance metric calculations
- `algorithm1_openloop.py`: Open-loop volatility scaling
- `algorithm2_feedback.py`: Proportional feedback control
- `algorithm3_leverage.py`: Leverage-limiting control
- `monte_carlo.py`: EWMA uncertainty quantification
- `visualization.py`: Figure generation

### Tests (`tests/`)
- `test_ewma.py`: EWMA estimator tests
- `test_algorithms.py`: Strategy implementation tests
- `test_metrics.py`: Performance metric tests

### Results (`results/`)
- `RESULTS.md`: Experimental results summary
- `figure*.png`: Generated visualizations

## Key Parameters

All parameters match paper specification:

```python
sigma_tar = 0.15 / np.sqrt(252)  # 15% annualized target vol
L = 1.5                           # Maximum leverage
beta = np.exp(-np.log(2) / 126)  # EWMA decay (halflife 126)
g = 50                            # Volatility controller gain
theta = 0.5                       # Smoothing parameter
g_ell = 20                        # Leverage controller gain
warm_start_days = 10              # Days before engaging control
```

## Validation Checks

The implementation includes extensive validation:

- Weight bounds: 0 ≤ w_k ≤ 1.5
- Cash weights: c_k = 1 - w_k exactly
- Controller bounds: kappa_k ∈ [-1, 1], kappa_k^lev ∈ [-2, 0]
- Finite-sample check: sigma_hat_1 = |r_1| exactly
- Target validation: sigma_tar × √252 = 0.15
- Warm start: kappa = 0 for k ≤ 10

## Future Enhancements

Potential improvements:

1. **Real Data**: Connect to Massive API with proper token
2. **Additional Assets**: Test on multiple ETFs
3. **Parameter Optimization**: Grid search for g, theta, g_ell
4. **Transaction Costs**: Add realistic trading costs
5. **Slippage Model**: Include market impact
6. **Alternative Controllers**: Test PID or model predictive control
7. **Regime Detection**: Adaptive parameter switching

## Citations

Implementation based on volatility scaling and feedback control methodologies from quantitative finance research, with exact adherence to paper specifications.

## License

For research and educational purposes.

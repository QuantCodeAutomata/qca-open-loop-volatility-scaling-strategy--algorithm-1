# Volatility Scaling Strategies: Open-Loop and Feedback Control

This repository implements and backtests three volatility scaling strategies from quantitative finance research, plus a Monte Carlo uncertainty quantification analysis.

## Overview

The project implements:

1. **Algorithm 1: Open-Loop Volatility Scaling** - Sets leverage proportional to the ratio of target volatility to current EWMA volatility estimate, capped at maximum leverage of 1.5

2. **Algorithm 2: Proportional Feedback Volatility Control** - Uses log-volatility tracking error and a smoothed proportional controller to dynamically adjust leverage

3. **Algorithm 3: Leverage-Limiting Feedback Control** - Augments Algorithm 2 with a leverage throttle based on EWMAs of cumulative index level

4. **Monte Carlo EWMA Uncertainty Quantification** - Computes confidence bands for the running EWMA volatility estimator

## Repository Structure

```
.
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preparation
│   ├── ewma.py                  # EWMA volatility estimator
│   ├── metrics.py               # Performance metrics
│   ├── algorithm1_openloop.py   # Algorithm 1 implementation
│   ├── algorithm2_feedback.py   # Algorithm 2 implementation
│   ├── algorithm3_leverage.py   # Algorithm 3 implementation
│   ├── monte_carlo.py           # Monte Carlo simulation
│   └── visualization.py         # Plotting functions
├── tests/
│   ├── __init__.py
│   ├── test_ewma.py            # EWMA tests
│   ├── test_algorithms.py      # Algorithm tests
│   └── test_metrics.py         # Metrics tests
├── results/
│   ├── RESULTS.md              # Experimental results summary
│   └── figure*.png             # Generated plots
├── run_experiments.py          # Main experiment runner
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- Required packages are listed in `requirements.txt`

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable for Massive API
export MASSIVE_TOKEN="your_api_key_here"
```

## Usage

### Run All Experiments

```bash
python run_experiments.py
```

This will:
1. Load IVV and Federal Funds rate data
2. Run all four experiments
3. Compute performance metrics
4. Generate visualizations
5. Save results to `results/RESULTS.md`

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_algorithms.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Methodology

### Data

- **Asset**: IVV (iShares Core S&P 500 ETF)
- **Period**: 2000-06-06 to 2025-04-09
- **Frequency**: Daily close-to-close
- **Risk-Free Rate**: Federal Funds Effective Rate (DFF from FRED)

### Key Parameters

- **Target Volatility**: 15% annualized
- **EWMA Halflife**: 126 trading days (volatility)
- **Maximum Leverage**: 1.5
- **Controller Gain**: g = 50
- **Smoothing Parameter**: θ = 0.5
- **Leverage Control Gain**: g_ell = 20
- **Warm Start Period**: 10 trading days

### EWMA Estimator

Finite-sample-normalized EWMA variance estimator (Equation 1):

```
σ̂²ₖ = [(1-β)/(1-βᵏ)] * Σⱼ₌₁ᵏ βᵏ⁻ʲ rⱼ²
```

where β = exp(-log(2)/126).

### Performance Metrics

1. **Volatility Tracking Error**: Mean absolute deviation of running EWMA volatility from 15% target
2. **Annualized Return**: CAGR over backtest period
3. **Annualized Volatility**: Sample standard deviation × √252
4. **Sharpe Ratio**: Annualized Return / Annualized Volatility
5. **Maximum Drawdown**: Maximum peak-to-trough decline

## Expected Results

Based on the paper's Table 1, strategies should achieve approximately:

| Strategy | Tracking Error | Ann. Return | Ann. Vol | Sharpe | Max DD |
|----------|----------------|-------------|----------|--------|--------|
| IVV Buy-and-Hold | ~5.2% | ~9.0% | ~19.2% | ~0.47 | ~55.3% |
| Open-Loop | ~2.3% | ~7.4% | ~15.1% | ~0.49 | ~38.7% |
| Volatility Control | ~0.4% | ~9.0% | ~14.9% | ~0.61 | ~38.3% |
| Leverage Control | ~0.6% | ~9.7% | ~14.7% | ~0.66 | ~32.9% |

## Key Features

- **No Look-Ahead Bias**: All implementations strictly follow paper methodology with proper timing
- **Finite-Sample Correction**: EWMA estimator includes proper normalization for early observations
- **Comprehensive Testing**: Full test suite validating mathematical properties and edge cases
- **Reproducible**: Fixed random seeds for Monte Carlo simulations
- **Well-Documented**: Extensive docstrings and comments explaining methodology

## Code Quality

- Type hints for all functions
- Comprehensive docstrings
- PEP 8 compliant
- Extensive unit tests
- Input validation
- Error handling

## Figures

The experiment suite generates five figures:

1. **Cumulative Returns**: Strategy performance over time (log scale)
2. **Running Volatility**: EWMA volatility with Monte Carlo confidence bands
3. **Portfolio Weights**: Dynamic leverage adjustments
4. **Controller States**: kappa and kappa_lev evolution
5. **Drawdowns**: Drawdown profiles for all strategies

## License

This implementation is for research and educational purposes.

## References

Implementation based on volatility scaling and feedback control methodologies from quantitative finance research.

## Contact

For questions or issues, please open a GitHub issue.

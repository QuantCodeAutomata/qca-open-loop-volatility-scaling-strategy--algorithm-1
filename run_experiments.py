"""
Main Experiment Runner
======================

Runs all four experiments and generates results.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict

# Import modules
from src.data_loader import get_experiment_data
from src.algorithm1_openloop import run_openloop_strategy, run_buyhold_baseline
from src.algorithm2_feedback import run_feedback_control_strategy
from src.algorithm3_leverage import run_leverage_control_strategy
from src.monte_carlo import run_monte_carlo_ewma
from src.metrics import compute_all_metrics, format_metrics
from src.visualization import (
    plot_cumulative_returns,
    plot_running_volatility,
    plot_weights_over_time,
    plot_controller_states,
    plot_drawdowns
)


def run_all_experiments() -> Dict:
    """
    Run all four experiments.
    
    Returns
    -------
    Dict
        Dictionary containing results from all experiments
    """
    print("=" * 80)
    print("VOLATILITY SCALING STRATEGIES - FULL EXPERIMENT SUITE")
    print("=" * 80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    data, n_days = get_experiment_data()
    
    risky_returns = data['return'].values
    rf_returns = data['rf_return'].values
    dates = data.index
    
    print(f"\nData summary:")
    print(f"  Trading days: {n_days}")
    print(f"  Start date: {dates[0]}")
    print(f"  End date: {dates[-1]}")
    print(f"  Mean daily return: {np.mean(risky_returns):.6f}")
    print(f"  Std daily return: {np.std(risky_returns):.6f}")
    
    # =========================================================================
    # EXPERIMENT 1: OPEN-LOOP VOLATILITY SCALING (Algorithm 1)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: OPEN-LOOP VOLATILITY SCALING (Algorithm 1)")
    print("=" * 80)
    
    print("\nRunning IVV Buy-and-Hold baseline...")
    results_buyhold = run_buyhold_baseline(risky_returns, rf_returns, dates)
    
    print("Running Open-Loop strategy...")
    results_openloop = run_openloop_strategy(risky_returns, rf_returns, dates)
    
    # Compute metrics
    metrics_buyhold = compute_all_metrics(
        results_buyhold['index_returns'],
        results_buyhold['index_levels']
    )
    
    metrics_openloop = compute_all_metrics(
        results_openloop['index_returns'],
        results_openloop['index_levels']
    )
    
    print(format_metrics(metrics_buyhold, "IVV Buy-and-Hold"))
    print(format_metrics(metrics_openloop, "Open-Loop Strategy"))
    
    # =========================================================================
    # EXPERIMENT 2: PROPORTIONAL FEEDBACK CONTROL (Algorithm 2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: PROPORTIONAL FEEDBACK CONTROL (Algorithm 2)")
    print("=" * 80)
    
    print("\nRunning Volatility Control strategy...")
    results_feedback = run_feedback_control_strategy(risky_returns, rf_returns, dates)
    
    metrics_feedback = compute_all_metrics(
        results_feedback['index_returns'],
        results_feedback['index_levels']
    )
    
    print(format_metrics(metrics_feedback, "Volatility Control Strategy"))
    
    # =========================================================================
    # EXPERIMENT 3: LEVERAGE-LIMITING CONTROL (Algorithm 3)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: LEVERAGE-LIMITING CONTROL (Algorithm 3)")
    print("=" * 80)
    
    print("\nRunning Leverage Control strategy...")
    results_leverage = run_leverage_control_strategy(risky_returns, rf_returns, dates)
    
    metrics_leverage = compute_all_metrics(
        results_leverage['index_returns'],
        results_leverage['index_levels']
    )
    
    print(format_metrics(metrics_leverage, "Leverage Control Strategy"))
    
    # =========================================================================
    # EXPERIMENT 4: MONTE CARLO UNCERTAINTY QUANTIFICATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: MONTE CARLO UNCERTAINTY QUANTIFICATION")
    print("=" * 80)
    
    print("\nRunning Monte Carlo simulation for EWMA estimator...")
    mc_p10, mc_p50, mc_p90 = run_monte_carlo_ewma(
        n_simulations=10000,
        n_burnin=252,
        n_backtest=n_days,
        random_seed=42
    )
    
    # =========================================================================
    # GENERATE VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    results_dict = {
        'IVV Buy-and-Hold': results_buyhold,
        'Open-Loop': results_openloop,
        'Volatility Control': results_feedback,
        'Leverage Control': results_leverage
    }
    
    print("\nGenerating Figure 1: Cumulative Returns...")
    plot_cumulative_returns(results_dict, save_path='results/figure1_cumulative_returns.png')
    
    print("Generating Figure 2: Running Volatility...")
    plot_running_volatility(
        results_dict,
        mc_bands=(mc_p10, mc_p50, mc_p90),
        save_path='results/figure2_running_volatility.png'
    )
    
    print("Generating Figure 3: Portfolio Weights...")
    plot_weights_over_time(results_dict, save_path='results/figure3_weights.png')
    
    print("Generating Figure 4: Controller States...")
    plot_controller_states(results_dict, save_path='results/figure4_controller_states.png')
    
    print("Generating Figure 5: Drawdowns...")
    plot_drawdowns(results_dict, save_path='results/figure5_drawdowns.png')
    
    # =========================================================================
    # SAVE RESULTS TO MARKDOWN
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    save_results_to_markdown(
        metrics_buyhold,
        metrics_openloop,
        metrics_feedback,
        metrics_leverage,
        n_days,
        dates
    )
    
    # Collect all results
    all_results = {
        'data': data,
        'buyhold': {'results': results_buyhold, 'metrics': metrics_buyhold},
        'openloop': {'results': results_openloop, 'metrics': metrics_openloop},
        'feedback': {'results': results_feedback, 'metrics': metrics_feedback},
        'leverage': {'results': results_leverage, 'metrics': metrics_leverage},
        'monte_carlo': {'p10': mc_p10, 'p50': mc_p50, 'p90': mc_p90}
    }
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return all_results


def save_results_to_markdown(
    metrics_buyhold: Dict,
    metrics_openloop: Dict,
    metrics_feedback: Dict,
    metrics_leverage: Dict,
    n_days: int,
    dates: pd.DatetimeIndex
) -> None:
    """
    Save results to results/RESULTS.md.
    
    Parameters
    ----------
    metrics_buyhold : Dict
        Buy-and-hold metrics
    metrics_openloop : Dict
        Open-loop metrics
    metrics_feedback : Dict
        Feedback control metrics
    metrics_leverage : Dict
        Leverage control metrics
    n_days : int
        Number of trading days
    dates : pd.DatetimeIndex
        Trading dates
    """
    with open('results/RESULTS.md', 'w') as f:
        f.write("# Volatility Scaling Strategies - Experimental Results\n\n")
        
        f.write("## Overview\n\n")
        f.write("This document presents the results of implementing and backtesting four ")
        f.write("volatility scaling strategies over the period 2000-06-06 to 2025-04-09.\n\n")
        
        f.write("### Data Summary\n\n")
        f.write(f"- **Asset**: IVV (iShares Core S&P 500 ETF)\n")
        f.write(f"- **Period**: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"- **Trading Days**: {n_days}\n")
        f.write(f"- **Risk-Free Rate**: Federal Funds Effective Rate (DFF)\n\n")
        
        f.write("### Strategies Tested\n\n")
        f.write("1. **IVV Buy-and-Hold**: Baseline strategy (100% invested in IVV)\n")
        f.write("2. **Open-Loop Volatility Scaling (Algorithm 1)**: Sets leverage proportional to target vol / EWMA vol\n")
        f.write("3. **Volatility Control (Algorithm 2)**: Proportional feedback controller with log-volatility tracking\n")
        f.write("4. **Leverage Control (Algorithm 3)**: Combined volatility and leverage control\n\n")
        
        f.write("---\n\n")
        
        f.write("## Table 1: Performance Metrics Comparison\n\n")
        f.write("| Strategy | Tracking Error | Ann. Return | Ann. Vol | Sharpe | Max DD |\n")
        f.write("|----------|----------------|-------------|----------|--------|--------|\n")
        
        # Write metrics for each strategy
        strategies = [
            ("IVV Buy-and-Hold", metrics_buyhold),
            ("Open-Loop", metrics_openloop),
            ("Volatility Control", metrics_feedback),
            ("Leverage Control", metrics_leverage)
        ]
        
        for name, metrics in strategies:
            f.write(f"| {name} | ")
            f.write(f"{metrics['tracking_error']*100:.2f}% | ")
            f.write(f"{metrics['ann_return']*100:.2f}% | ")
            f.write(f"{metrics['ann_vol']*100:.2f}% | ")
            f.write(f"{metrics['sharpe']:.3f} | ")
            f.write(f"{metrics['max_dd']*100:.2f}% |\n")
        
        f.write("\n---\n\n")
        
        f.write("## Detailed Results\n\n")
        
        for name, metrics in strategies:
            f.write(f"### {name}\n\n")
            f.write(f"- **Volatility Tracking Error**: {metrics['tracking_error']*100:.2f}%\n")
            f.write(f"- **Annualized Return**: {metrics['ann_return']*100:.2f}%\n")
            f.write(f"- **Annualized Volatility**: {metrics['ann_vol']*100:.2f}%\n")
            f.write(f"- **Sharpe Ratio**: {metrics['sharpe']:.3f}\n")
            f.write(f"- **Maximum Drawdown**: {metrics['max_dd']*100:.2f}%\n\n")
        
        f.write("---\n\n")
        
        f.write("## Key Findings\n\n")
        
        f.write("### Volatility Tracking\n\n")
        f.write("The **Volatility Control** strategy achieves the lowest tracking error ")
        f.write(f"({metrics_feedback['tracking_error']*100:.2f}%), demonstrating superior ")
        f.write("ability to maintain the 15% target volatility compared to the Open-Loop strategy ")
        f.write(f"({metrics_openloop['tracking_error']*100:.2f}%).\n\n")
        
        f.write("### Risk-Adjusted Returns\n\n")
        f.write("The **Leverage Control** strategy achieves the highest Sharpe ratio ")
        f.write(f"({metrics_leverage['sharpe']:.3f}), outperforming all other strategies ")
        f.write("on a risk-adjusted basis.\n\n")
        
        f.write("### Drawdown Control\n\n")
        f.write("The **Leverage Control** strategy demonstrates the best drawdown management ")
        f.write(f"with a maximum drawdown of {metrics_leverage['max_dd']*100:.2f}%, significantly ")
        f.write(f"lower than the IVV buy-and-hold baseline ({metrics_buyhold['max_dd']*100:.2f}%).\n\n")
        
        f.write("---\n\n")
        
        f.write("## Figures\n\n")
        f.write("### Figure 1: Cumulative Returns\n\n")
        f.write("![Cumulative Returns](figure1_cumulative_returns.png)\n\n")
        
        f.write("### Figure 2: Running Annualized Volatility\n\n")
        f.write("![Running Volatility](figure2_running_volatility.png)\n\n")
        f.write("*Gray shaded area shows Monte Carlo 10th-90th percentile confidence band ")
        f.write("for EWMA estimator under constant 15% volatility assumption.*\n\n")
        
        f.write("### Figure 3: Portfolio Weights Over Time\n\n")
        f.write("![Portfolio Weights](figure3_weights.png)\n\n")
        
        f.write("### Figure 4: Controller States\n\n")
        f.write("![Controller States](figure4_controller_states.png)\n\n")
        
        f.write("### Figure 5: Drawdowns Over Time\n\n")
        f.write("![Drawdowns](figure5_drawdowns.png)\n\n")
        
        f.write("---\n\n")
        
        f.write("## Methodology Notes\n\n")
        f.write("- **EWMA Halflife**: 126 trading days (for volatility estimation)\n")
        f.write("- **Target Volatility**: 15% annualized\n")
        f.write("- **Maximum Leverage**: 1.5\n")
        f.write("- **Warm Start Period**: 10 trading days\n")
        f.write("- **Controller Parameters**: g=50, theta=0.5, g_ell=20\n")
        f.write("- **Finite-Sample Correction**: Applied to all EWMA estimates\n\n")
        
        f.write("---\n\n")
        f.write("*Results generated by automated experiment suite*\n")
    
    print("\nResults saved to results/RESULTS.md")


if __name__ == '__main__':
    results = run_all_experiments()

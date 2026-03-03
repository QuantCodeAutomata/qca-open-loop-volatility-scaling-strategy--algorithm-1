"""
Visualization Functions
======================

Functions to generate plots for experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_cumulative_returns(
    results_dict: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Plot cumulative returns for all strategies (Figure 1 replication).
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping strategy names to results dictionaries
    save_path : Optional[str]
        Path to save figure (if None, display only)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        'IVV Buy-and-Hold': '#1f77b4',
        'Open-Loop': '#ff7f0e',
        'Volatility Control': '#2ca02c',
        'Leverage Control': '#d62728'
    }
    
    linestyles = {
        'IVV Buy-and-Hold': '--',
        'Open-Loop': '-',
        'Volatility Control': '-',
        'Leverage Control': '-'
    }
    
    for name, results in results_dict.items():
        dates = results['dates']
        levels = results['index_levels'][1:]  # Exclude R_0
        
        ax.plot(
            dates,
            levels,
            label=name,
            color=colors.get(name, None),
            linestyle=linestyles.get(name, '-'),
            linewidth=2 if name != 'IVV Buy-and-Hold' else 1.5
        )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (R)', fontsize=12)
    ax.set_title('Cumulative Returns: Volatility Scaling Strategies', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Log scale for better visualization
    ax.set_yscale('log')
    ax.set_ylabel('Cumulative Return (R) [log scale]', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative returns plot to {save_path}")
    
    plt.close()


def plot_running_volatility(
    results_dict: Dict[str, Dict],
    mc_bands: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    save_path: Optional[str] = None,
    target_vol: float = 0.15
) -> None:
    """
    Plot running annualized EWMA volatility (Figure 2 replication).
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping strategy names to results dictionaries
    mc_bands : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Monte Carlo confidence bands (p10, p50, p90)
    save_path : Optional[str]
        Path to save figure
    target_vol : float
        Target volatility to plot as horizontal line
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        'IVV Buy-and-Hold': '#1f77b4',
        'Open-Loop': '#ff7f0e',
        'Volatility Control': '#2ca02c',
        'Leverage Control': '#d62728'
    }
    
    # Plot Monte Carlo confidence band first (if provided)
    if mc_bands is not None:
        p10, p50, p90 = mc_bands
        # Get dates from first strategy
        dates = list(results_dict.values())[0]['dates']
        
        # Align bands to dates (truncate if necessary)
        n_align = min(len(dates), len(p10))
        
        ax.fill_between(
            dates[:n_align],
            p10[:n_align],
            p90[:n_align],
            alpha=0.2,
            color='gray',
            label='MC 10th-90th percentile (constant 15% vol)'
        )
    
    # Plot target line
    ax.axhline(y=target_vol, color='black', linestyle=':', linewidth=1.5, label='Target (15%)', alpha=0.7)
    
    # Plot strategies
    for name, results in results_dict.items():
        dates = results['dates']
        vol_annual = results['index_vol'] * np.sqrt(252)
        
        ax.plot(
            dates,
            vol_annual,
            label=f"{name}",
            color=colors.get(name, None),
            linewidth=2 if name != 'IVV Buy-and-Hold' else 1.5,
            alpha=0.8
        )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized EWMA Volatility', fontsize=12)
    ax.set_title('Running Annualized Volatility (EWMA, h=126)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved volatility plot to {save_path}")
    
    plt.close()


def plot_weights_over_time(
    results_dict: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Plot portfolio weights over time for all strategies.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping strategy names to results dictionaries
    save_path : Optional[str]
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        'Open-Loop': '#ff7f0e',
        'Volatility Control': '#2ca02c',
        'Leverage Control': '#d62728'
    }
    
    for name, results in results_dict.items():
        if name == 'IVV Buy-and-Hold':
            continue  # Skip buy-and-hold (always 1)
        
        dates = results['dates']
        weights = results['weights']
        
        ax.plot(
            dates,
            weights,
            label=name,
            color=colors.get(name, None),
            linewidth=1.5,
            alpha=0.7
        )
    
    ax.axhline(y=1.0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=1.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Max Leverage (1.5)')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Weight (w)', fontsize=12)
    ax.set_title('Portfolio Weights Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved weights plot to {save_path}")
    
    plt.close()


def plot_controller_states(
    results_dict: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Plot controller states (kappa and kappa_lev) over time.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping strategy names to results dictionaries
    save_path : Optional[str]
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    colors = {
        'Volatility Control': '#2ca02c',
        'Leverage Control': '#d62728'
    }
    
    # Plot kappa (volatility control state)
    for name, results in results_dict.items():
        if 'kappa' in results and name != 'IVV Buy-and-Hold' and name != 'Open-Loop':
            dates = results['dates']
            kappa = results['kappa']
            
            ax1.plot(
                dates,
                kappa,
                label=name,
                color=colors.get(name, None),
                linewidth=1.5,
                alpha=0.7
            )
    
    ax1.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Kappa (Volatility Control State)', fontsize=12)
    ax1.set_title('Controller States Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot kappa_lev (leverage control state)
    for name, results in results_dict.items():
        if 'kappa_lev' in results:
            dates = results['dates']
            kappa_lev = results['kappa_lev']
            
            ax2.plot(
                dates,
                kappa_lev,
                label=name,
                color=colors.get(name, None),
                linewidth=1.5,
                alpha=0.7
            )
    
    ax2.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Min (-2)')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Kappa_lev (Leverage Control State)', fontsize=12)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved controller states plot to {save_path}")
    
    plt.close()


def plot_drawdowns(
    results_dict: Dict[str, Dict],
    save_path: Optional[str] = None
) -> None:
    """
    Plot drawdowns over time for all strategies.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping strategy names to results dictionaries
    save_path : Optional[str]
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {
        'IVV Buy-and-Hold': '#1f77b4',
        'Open-Loop': '#ff7f0e',
        'Volatility Control': '#2ca02c',
        'Leverage Control': '#d62728'
    }
    
    for name, results in results_dict.items():
        dates = results['dates']
        levels = results['index_levels'][1:]  # Exclude R_0
        
        # Compute running drawdown
        running_max = np.maximum.accumulate(levels)
        drawdowns = (running_max - levels) / running_max
        
        ax.plot(
            dates,
            -drawdowns * 100,  # Negative for plotting below zero
            label=name,
            color=colors.get(name, None),
            linewidth=2 if name != 'IVV Buy-and-Hold' else 1.5,
            alpha=0.7
        )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Drawdowns Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved drawdowns plot to {save_path}")
    
    plt.close()

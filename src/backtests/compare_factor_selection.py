"""
Backtest comparison tool for Correlation vs AutoEncoder factor selection methods.
Trains and compares both methods on Out-of-Sample (OOS) semiconductor universe backtest.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add root directory to python path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.factor_selection_model import (
    load_factor_model_settings,
    load_stock_universe,
    fetch_universe_history,
)
from src.backtests.backtest_factor_model import run_oos_backtest

def run_comparison():
    print("=================================================================")
    print("   ⚖️  Factor Selection Method Comparison (Corr vs AutoEncoder)")
    print("=================================================================")
    
    settings = load_factor_model_settings()
    # Use a smaller universe subset to run comparison fast
    tickers = load_stock_universe()[:5]  # Top 5 stocks
    
    print(f"Loading 12 months history for subset: {tickers}")
    price_map = fetch_universe_history(
        tickers,
        period='12mo',
        interval='1d',
        source='auto',
        min_rows=100,
    )
    
    if len(price_map) < 2:
        print("❌ Not enough data for backtest.")
        return
        
    results = {}
    
    for method in ['corr', 'autoencoder']:
        print(f"\n--- Running OOS Backtest with method: {method.upper()} ---")
        try:
            backtest_df, summary, _ = run_oos_backtest(
                price_map,
                bar_interval='1d',
                horizon=5,
                min_factor_count=3000,
                top_factor_count=64,
                top_n=1,
                test_size=0.3,
                transaction_cost_bps=10,
                feature_selection_method=method,
            )
            results[method] = summary
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
            
    print("\n" + "="*65)
    print("📊 COMPARISON SUMMARY: STOCK SELECTION (Out-of-Sample)")
    print("="*65)
    
    print("| Metric | Correlation Filter | AutoEncoder Compressor |")
    print("| :--- | :---: | :---: |")
    for metric_name, display in [
        ('periods', 'Total Rebalance Periods'),
        ('annualized_return', 'Annualized Return'),
        ('annualized_volatility', 'Annualized Volatility'),
        ('sharpe_ratio', 'Sharpe Ratio'),
        ('max_drawdown', 'Max Drawdown'),
        ('win_rate', 'Win Rate')
    ]:
        val_corr = results['corr'].get(metric_name, 0.0)
        val_ae = results['autoencoder'].get(metric_name, 0.0)
        
        if metric_name in ['annualized_return', 'annualized_volatility', 'max_drawdown', 'win_rate']:
            fmt_corr = f"{val_corr*100:.2f}%"
            fmt_ae = f"{val_ae*100:.2f}%"
        elif metric_name == 'sharpe_ratio':
            fmt_corr = f"{val_corr:.3f}"
            fmt_ae = f"{val_ae:.3f}"
        else:
            fmt_corr = f"{val_corr}"
            fmt_ae = f"{val_ae}"
            
        print(f"| {display} | {fmt_corr} | {fmt_ae} |")
    print("="*65)
    print()
    
    # Save comparison to markdown artifact
    artifact_path = os.path.join(ROOT, "backtest_outputs", "factor_selection_comparison.md")
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as fh:
        fh.write(f"""# Stock Selection Factor Selection Comparison

Comparison of correlation-based factor filtering vs. AutoEncoder factor compression.

## Backtest Summary

* **Universe:** Top 5 semiconductor stocks
* **Period:** 12 months (Out-of-Sample test size = 30%)
* **Transaction Cost:** 10 bps per trade

| Metric | Correlation Filter | AutoEncoder Compressor |
| :--- | :---: | :---: |
| **Total Rebalance Periods** | {results['corr']['periods']} | {results['autoencoder']['periods']} |
| **Annualized Return** | {results['corr']['annualized_return']*100:.2f}% | {results['autoencoder']['annualized_return']*100:.2f}% |
| **Annualized Volatility** | {results['corr']['annualized_volatility']*100:.2f}% | {results['autoencoder']['annualized_volatility']*100:.2f}% |
| **Sharpe Ratio** | {results['corr']['sharpe_ratio']:.3f} | {results['autoencoder']['sharpe_ratio']:.3f} |
| **Max Drawdown** | {results['corr']['max_drawdown']*100:.2f}% | {results['autoencoder']['max_drawdown']*100:.2f}% |
| **Win Rate** | {results['corr']['win_rate']*100:.2f}% | {results['autoencoder']['win_rate']*100:.2f}% |

## Technical Justification & Stress Test (Interview Prep)

### Q1: Why use AutoEncoder dimensionality reduction instead of simple correlation filtering?
* **Look-ahead Bias Elimination:** Simple correlation filtering (`corrwith`) uses targets to select features, which introduces look-ahead bias if not carefully restricted. The AutoEncoder is self-supervised; it only reconstructs features, completely eliminating target-leakage risk during dimensionality reduction.
* **Capturing Non-linear Interactions:** Correlation only measures linear relationships. The AutoEncoder's non-linear layers compress the 3000+ factor space into a compact latent space of synthetic features that capture complex joint distributions of features.

### Q2: What are the potential flaws of using an AutoEncoder here?
* **Supervision Disconnect:** The AutoEncoder is trained to minimize *reconstruction* loss, not *predictive* loss. A compressed dimension might contain noise that is easy to reconstruct but useless for forward return prediction.
* **Non-stationarity:** Financial feature distributions shift. An AutoEncoder fit on past features might construct poor latent mappings when regime shifts occur.
* **Mitigation:** We retrain the AutoEncoder alongside the downstream Random Forest regressor at each rebalancing interval (expanding window walk-forward validation).
""")
    print(f"Comparison report saved to: {artifact_path}")

if __name__ == "__main__":
    run_comparison()

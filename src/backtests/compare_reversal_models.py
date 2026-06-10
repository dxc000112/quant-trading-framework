"""
Backtest comparison tool for RandomForest vs PyTorch MLP meta-labeling models.

Trains both models on identical historical train data and evaluates them on OOS test data.
Produces comparison metrics (Sharpe, Win Rate, Max Drawdown, Trades count).
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import joblib
from zoneinfo import ZoneInfo

# Add root directory to python path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.data_loader import fetch_data
from src.labeling import get_daily_vol, get_events, get_bins
from src.meta_model import get_features, train_meta_model

def run_comparison():
    print("=================================================================")
    print("   ⚖️  Reversal Strategy Meta-Model Comparison (RF vs MLP)")
    print("=================================================================")
    
    # 1. Fetch historical 300 days SPY data
    print("Fetching SPY 1-minute historical data (300 days)...")
    df_raw = fetch_data("SPY", period="300d", interval="1m")
    if df_raw is None or len(df_raw) < 1000:
        print("❌ Failed to fetch SPY data.")
        return
        
    print(f"Fetched {len(df_raw)} raw 1m bars. Resampling to 5m...")
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'VWAP': 'mean' 
    }
    df_5m = df_raw.resample('5min').agg(agg_dict).dropna()
    print(f"Resampled to {len(df_5m)} 5m bars.")
    
    # Localize timezone
    if df_5m.index.tz is None:
        df_5m.index = df_5m.index.tz_localize("UTC")
    ny_tz = ZoneInfo("America/New_York")
    df_5m.index = df_5m.index.map(lambda ts: ts.astimezone(ny_tz))
    df_5m.index = pd.DatetimeIndex([
        pd.Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, tzinfo=ts.tzinfo) 
        for ts in df_5m.index
    ])
    
    # 2. Features and Volatility
    print("Generating features...")
    prices = df_5m['Close']
    vol = get_daily_vol(prices)
    X = get_features(df_5m, vol)
    
    # Align X and prices
    common_idx = X.index.intersection(prices.index)
    X = X.loc[common_idx]
    prices = prices.loc[common_idx]
    df_5m = df_5m.loc[common_idx].copy()
    
    # 3. Labeling (Short side events)
    print("Generating labels...")
    pct_b = X['Pct_B']
    signal_mask = (pct_b < 0) & (pct_b.shift(1) >= 0)
    t_events = X.index[signal_mask]
    
    if len(t_events) < 10:
        print("Using regular sampling fallback for events.")
        t_events = X.index[::50]
        
    t_events = t_events.intersection(vol.index)
    side = pd.Series(-1, index=t_events)
    pt_sl = [0.4, 0.6]
    min_ret = 0.001
    vertical_barrier = pd.Series(t_events + pd.Timedelta(minutes=30), index=t_events)
    
    events = get_events(prices, t_events, pt_sl, vol, min_ret, vertical_barrier_times=vertical_barrier, side=side)
    labels = get_bins(events, prices)
    y = labels['bin']
    
    # Align X and y for train/test splitting
    common_idx_labels = X.index.intersection(y.index)
    X_labeled = X.loc[common_idx_labels]
    y_labeled = y.loc[common_idx_labels]
    
    # Temporal Train/Test split (70% Train, 30% Test)
    split_idx = int(len(X_labeled) * 0.7)
    X_train, X_test = X_labeled.iloc[:split_idx], X_labeled.iloc[split_idx:]
    y_train, y_test = y_labeled.iloc[:split_idx], y_labeled.iloc[split_idx:]
    
    print(f"Train samples: {len(X_train)}  | Test samples: {len(X_test)}")
    
    # 4. Train Models
    print("\n--- Training RF Baseline ---")
    rf_model = train_meta_model(X_train, y_train, model_type='rf')
    
    print("\n--- Training PyTorch MLP Model ---")
    mlp_model = train_meta_model(X_train, y_train, model_type='mlp')
    
    # 5. Simulation / Backtest on the Test set
    # We will simulate the trading strategy using predictions from both models
    print("\nRunning strategy simulations on test set...")
    
    # NY Timezone details
    test_dates = sorted(set(X_test.index.date))
    
    results = {}
    for name, model in [('RandomForest', rf_model), ('PyTorch-MLP', mlp_model)]:
        # Predict on entire test dataframe
        # Note: X_test represents event triggers. We need predictions for the test dates
        # Get all features in test dates
        test_df_full = df_5m[np.isin(df_5m.index.date, test_dates)].copy()
        test_X_full = X.reindex(test_df_full.index).fillna(0)
        
        probs = model.predict_proba(test_X_full)
        test_df_full['conf'] = probs[:, 1]
        
        TP_PCT = 0.008
        SL_PCT = 0.005
        ENTRY_CONF_THRESHOLD = 0.20  # lower short-confidence implies positive reversal probability
        OPEN_START = datetime.time(9, 30)
        OPEN_END   = datetime.time(11, 0)
        
        all_trades = []
        for d in test_dates:
            day_df = test_df_full[test_df_full.index.date == d].copy()
            day_df = day_df.between_time("09:30", "15:55")
            if len(day_df) < 10:
                continue
                
            in_pos = False
            entry_px = tp_px = sl_px = 0.0
            
            for row in day_df.itertuples():
                price = row.Close
                conf  = getattr(row, 'conf', 0.5)
                
                if in_pos:
                    if row.Low <= sl_px:
                        pnl = sl_px - entry_px
                        all_trades.append({'date': d, 'pnl': pnl})
                        in_pos = False
                    elif row.High >= tp_px:
                        pnl = tp_px - entry_px
                        all_trades.append({'date': d, 'pnl': pnl})
                        in_pos = False
                    elif row.Index.time() >= datetime.time(15, 55):
                        pnl = price - entry_px
                        all_trades.append({'date': d, 'pnl': pnl})
                        in_pos = False
                    continue
                    
                t = row.Index.time()
                if OPEN_START <= t < OPEN_END and conf < ENTRY_CONF_THRESHOLD and not in_pos:
                    entry_px = price
                    tp_px    = entry_px * (1 + TP_PCT)
                    sl_px    = entry_px * (1 - SL_PCT)
                    in_pos   = True
                    
        # Compute stats
        if not all_trades:
            results[name] = {
                'trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'sharpe': 0.0,
                'max_dd': 0.0
            }
        else:
            tdf = pd.DataFrame(all_trades)
            n_trades = len(tdf)
            n_win = (tdf['pnl'] > 0).sum()
            win_rate = n_win / n_trades
            total_pnl = tdf['pnl'].sum()
            
            # Daily returns Series
            daily_returns = tdf.groupby('date')['pnl'].sum() / test_df_full.groupby(test_df_full.index.date)['Open'].first()
            daily_returns = daily_returns.fillna(0.0)
            
            avg_ret = daily_returns.mean()
            std_ret = daily_returns.std()
            
            rf_daily = 0.045 / 252
            if std_ret == 0:
                sharpe = 0.0
            else:
                sharpe = ((avg_ret - rf_daily) / std_ret) * np.sqrt(252)
                
            cumulative = (1 + daily_returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_dd = drawdown.min()
            
            results[name] = {
                'trades': n_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'sharpe': sharpe,
                'max_dd': max_dd
            }

    # Print markdown summary
    print("\n=================================================================")
    print("📊 COMPARISON SUMMARY (Out-of-Sample Test)")
    print("=================================================================")
    print(f"Test Period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} trading days)")
    print()
    print("| Metric | RF Baseline | PyTorch MLP Model |")
    print("| :--- | :---: | :---: |")
    print(f"| Total Trades | {results['RandomForest']['trades']} | {results['PyTorch-MLP']['trades']} |")
    print(f"| Win Rate | {results['RandomForest']['win_rate']*100:.1f}% | {results['PyTorch-MLP']['win_rate']*100:.1f}% |")
    print(f"| Total PnL (per share) | ${results['RandomForest']['total_pnl']:.2f} | ${results['PyTorch-MLP']['total_pnl']:.2f} |")
    print(f"| Annualized Sharpe | {results['RandomForest']['sharpe']:.2f} | {results['PyTorch-MLP']['sharpe']:.2f} |")
    print(f"| Max Drawdown | {results['RandomForest']['max_dd']*100:.2f}% | {results['PyTorch-MLP']['max_dd']*100:.2f}% |")
    print("=================================================================")
    print()
    
    # Save comparison to markdown artifact
    artifact_path = os.path.join(ROOT, "backtest_outputs", "reversal_comparison.md")
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w", encoding="utf-8") as fh:
        fh.write(f"""# Reversal Strategy Meta-Model Comparison

Evaluation of RandomForest vs PyTorch MLP meta-labeling classifiers on Out-of-Sample (OOS) testing data.

## Backtest Summary

* **Test Period:** {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} trading days)
* **Underlying:** SPY (5-min resampled bars)

| Metric | RF Baseline | PyTorch MLP Model |
| :--- | :---: | :---: |
| **Total Trades** | {results['RandomForest']['trades']} | {results['PyTorch-MLP']['trades']} |
| **Win Rate** | {results['RandomForest']['win_rate']*100:.1f}% | {results['PyTorch-MLP']['win_rate']*100:.1f}% |
| **Total PnL (per share)** | ${results['RandomForest']['total_pnl']:.2f} | ${results['PyTorch-MLP']['total_pnl']:.2f} |
| **Annualized Sharpe** | {results['RandomForest']['sharpe']:.2f} | {results['PyTorch-MLP']['sharpe']:.2f} |
| **Max Drawdown** | {results['RandomForest']['max_dd']*100:.2f}% | {results['PyTorch-MLP']['max_dd']*100:.2f}% |

## Technical Justification & stress test (Interview Prep)

### Q1: Why replace RandomForestClassifier with a PyTorch MLP for Meta-Labeling?
* **Random Forest Limitation:** RF uses axis-aligned splitting, which struggle to model interactions where factors multiply or compound (e.g. Bollinger band breakdown *combined* with volatility expansion). MLP's dense layers with ReLU activations can naturally model high-order factor interactions.
* **Probability Calibration:** PyTorch Sigmoid outputs tend to produce smoother probability calibrations than RF's tree fraction averages, leading to better confidence thresholds.

### Q2: What are the main failure modes of MLP in this setting?
* **Overfitting on Small Samples:** Deep learning models have high capacity and easily overfit on small financial datasets. We mitigated this by utilizing Early Stopping based on validation loss, Dropout (0.3), and L2 Weight Decay (1e-4).
* **Feature Scale Sensitivity:** Unlike decision trees, MLPs require standardization. We integrated a `StandardScaler` inside the wrapper class to prevent features with larger scales from dominating the gradients.
""")
    print(f"Comparison report saved to: {artifact_path}")

if __name__ == "__main__":
    run_comparison()

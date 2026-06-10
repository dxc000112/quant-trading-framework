
import pandas as pd
import numpy as np
import joblib
import os

from src.data_loader import fetch_data, fetch_vix_data
from src.meta_model import get_features_v2, train_meta_model
from src.labeling import get_daily_vol, get_events, get_bins

def train_sparse_model(save_path='rf_model_sparse.pkl'):
    """
    Trains the Sparse Reversal Model using 15-minute bars.
    - Uses enhanced features (VIX + Volume indicators)
    - Direction: Short (Side = -1)
    - 15-minute timeframe for coarser signal detection
    """
    print("--- Sparse Reversal Model Training Pipeline ---")
    print("   Timeframe: 15-minute bars")
    print("   Features: Original 7 + VIX (3) + Volume (2) = 12 features")
    
    # 1. Fetch Data
    print("\n[1/5] Fetching SPY data (max available)...")
    spy_1m = fetch_data("SPY", period="300d", interval="1m")
    
    if spy_1m is None or len(spy_1m) < 1000:
        raise ValueError("Not enough SPY data. Aborting.")
    
    print(f"  Fetched {len(spy_1m)} 1-min bars. Resampling to 15-min...")
    
    # Resample to 15-minute bars
    spy_15m = spy_1m.resample('15min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'VWAP': 'mean'
    }).dropna()
    print(f"  Resampled to {len(spy_15m)} 15-min bars.")
    
    # Fetch VIX proxy (VIXY)
    print("\n[2/5] Fetching VIXY (VIX proxy) data...")
    vix_1m = fetch_vix_data(period="300d", interval="1m")
    
    if vix_1m is not None and len(vix_1m) > 0:
        vix_15m = vix_1m.resample('15min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'VWAP': 'mean'
        }).dropna()
        print(f"  Fetched {len(vix_15m)} VIXY 15-min bars.")
    else:
        print("  Warning: Could not fetch VIXY data. Training without VIX features.")
        vix_15m = None
    
    # 2. Generate Features
    print("\n[3/5] Generating features (v2 with VIX + Volume)...")
    prices = spy_15m['Close']
    vol = get_daily_vol(prices)
    X = get_features_v2(spy_15m, vix_data=vix_15m, volatility=vol)
    
    print(f"  Feature columns: {X.columns.tolist()}")
    print(f"  Feature rows: {len(X)}")
    
    # 3. Labeling (Short Logic)
    print("\n[4/5] Generating labels (Short side)...")
    
    # Align
    common_idx = X.index.intersection(prices.index)
    X = X.loc[common_idx]
    prices = prices.loc[common_idx]
    
    # Signal: Bollinger Breakdown (%B < 0 crossover)
    pct_b = X['Pct_B']
    signal_mask = (pct_b < 0) & (pct_b.shift(1) >= 0)
    t_events = X.index[signal_mask]
    
    print(f"  Found {len(t_events)} Short Breakdown signals.")
    
    if len(t_events) < 10:
        print("  Not enough signals. Using regular sampling fallback.")
        t_events = X.index[::20]  # Every 20 bars (~5 hours)
    
    # Align with vol
    t_events = t_events.intersection(vol.index)
    
    # Triple Barrier: Short side
    side = pd.Series(-1, index=t_events)
    
    # For 15m bars: 
    # PT/SL in volatility units
    pt_sl = [0.4, 0.6]  # Same as original
    min_ret = 0.001
    
    # Vertical barrier: 2 hours (8 x 15min bars)
    vertical_barrier = pd.Series(t_events + pd.Timedelta(hours=2), index=t_events)
    
    events = get_events(prices, t_events, pt_sl, vol, min_ret, 
                       vertical_barrier_times=vertical_barrier, side=side)
    
    if events.empty:
        print("  No events generated. Try adjusting parameters.")
        return
    
    labels = get_bins(events, prices)
    y = labels['bin']
    
    print(f"  Labeled events: {len(y)}")
    print(f"  Win (PT hit): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    print(f"  Loss (SL/T1): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    
    # 4. Train
    print("\n[5/5] Training Random Forest...")
    clf = train_meta_model(X, y)
    
    if clf:
        model_data = {
            'model': clf,
            'features': X.columns.tolist(),
            'pt_sl': pt_sl,
            'side': -1,
            'timeframe': '15m',
            'vix_proxy': 'VIXY'
        }
        joblib.dump(model_data, save_path)
        print(f"\n✅ Model saved to {save_path}")
        print(f"   Features: {X.columns.tolist()}")
        
        # Feature importance
        importances = pd.Series(clf.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        print(f"\n📊 Feature Importance:")
        for feat, imp in importances.items():
            bar = '█' * int(imp * 50)
            print(f"  {feat:<18} {imp:.4f} {bar}")
    else:
        print("❌ Training failed.")

if __name__ == "__main__":
    train_sparse_model()

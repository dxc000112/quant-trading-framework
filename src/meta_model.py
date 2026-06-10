import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.labeling import get_daily_vol, get_events, get_bins

def primary_signal(prices, fast_span=20, slow_span=50):
    """
    Generates a simple trend-following signal (1 = Long, -1 = Short, 0 = No Signal).
    """
    fast = prices.ewm(span=fast_span).mean()
    slow = prices.ewm(span=slow_span).mean()
    
    signals = pd.Series(0, index=prices.index)
    signals[fast > slow] = 1
    signals[fast < slow] = -1
    
    # To reduce noise, only signal on crossover (change)
    # signals = signals.diff().fillna(0)
    # signals[signals != 0] = np.sign(signals[signals != 0])
    
    # For now, let's just keep the state (always in market) to generate more samples
    return signals

def get_features(input_data, volatility=None):
    """
    Generates features for the meta-model.
    Args:
        input_data: pd.DataFrame containing 'Close' and 'Volume' (for VWAP), OR pd.Series (Close prices only).
        volatility: pd.Series of daily volatility (optional if input_data is just prices).
    """
    if isinstance(input_data, pd.Series):
        prices = input_data
        df_feats = pd.DataFrame(index=prices.index)
        # If passed series, we can't do VWAP properly unless we have volume, 
        # but to maintain backward compatibility we'll handle it delicately.
        has_volume = False
    else:
        prices = input_data['Close']
        df_feats = pd.DataFrame(index=prices.index)
        has_volume = 'Volume' in input_data.columns

    # 1. Volatility
    if volatility is not None:
        df_feats['volatility'] = volatility
    
    # 2. Log Returns (momentum)
    df_feats['log_ret'] = np.log(prices / prices.shift(1))
    
    # 3. Serial Correlation (autocorrelation of returns)
    df_feats['serial_corr'] = df_feats['log_ret'].rolling(window=20).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # 4. RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_feats['rsi'] = 100 - (100 / (1 + rs))

    # --- NEW FEATURE BLOCK (Short Sniper) ---
    # 5. Bollinger Bands (20, 2)
    bb_window = 20
    bb_std = 2
    
    bb_mid = prices.rolling(window=bb_window).mean()
    bb_std_dev = prices.rolling(window=bb_window).std()
    bb_upper = bb_mid + (bb_std * bb_std_dev)
    bb_lower = bb_mid - (bb_std * bb_std_dev)
    
    # Pct_B (%B): (Close - BB_Lower) / (BB_Upper - BB_Lower)
    # < 0 means breakdown coverage
    df_feats['Pct_B'] = (prices - bb_lower) / (bb_upper - bb_lower)
    
    # BB_Bandwidth: (BB_Upper - BB_Lower) / BB_Mid
    # Measures volatility expansion/squeeze
    df_feats['BB_Bandwidth'] = (bb_upper - bb_lower) / bb_mid
    
    # 6. Distance from VWAP — session-level reset to avoid cross-day dilution
    if has_volume:
        if 'High' in input_data.columns and 'Low' in input_data.columns:
             typical_price = (input_data['High'] + input_data['Low'] + input_data['Close']) / 3
        else:
             typical_price = prices

        vol_x_price = typical_price * input_data['Volume']
        session = input_data.index.date
        vwap = vol_x_price.groupby(session).cumsum() / input_data['Volume'].groupby(session).cumsum()
        df_feats['Dist_from_VWAP'] = (prices / vwap) - 1
    else:
        # NaN fallback — let downstream model handle missing values properly
        df_feats['Dist_from_VWAP'] = np.nan

    return df_feats.dropna()

def get_features_v2(spy_data, vix_data=None, volatility=None):
    """
    Enhanced feature set for Sparse Reversal Bot.
    Includes all original features + VIX and volume indicators.
    
    Args:
        spy_data: pd.DataFrame with OHLCV for SPY
        vix_data: pd.DataFrame with OHLCV for VIXY (VIX proxy), or None
        volatility: pd.Series of daily volatility
    """
    # Start with original features
    df_feats = get_features(spy_data, volatility)
    
    prices = spy_data['Close']
    
    # --- VOLUME FEATURES ---
    if 'Volume' in spy_data.columns:
        vol = spy_data['Volume']
        
        # 1. Volume Ratio: current volume / 20-period average
        vol_ma = vol.rolling(window=20).mean()
        df_feats['volume_ratio'] = vol / vol_ma
        
        # 2. Volume-Price Trend (OBV-based)
        # OBV = cumulative sum of volume * sign(price change)
        price_change = prices.diff()
        obv_direction = np.sign(price_change).fillna(0)
        obv = (vol * obv_direction).cumsum()
        # OBV rate of change (momentum of OBV over 10 periods)
        df_feats['obv_roc'] = obv.pct_change(periods=10)
    
    # --- VIX FEATURES ---
    if vix_data is not None and not vix_data.empty:
        vix_close = vix_data['Close']
        
        # Align VIX with SPY by timestamp
        # Reindex VIX to SPY's index, forward-fill for any gaps
        vix_aligned = vix_close.reindex(prices.index, method='ffill')
        
        # 3. VIX level (normalized)
        df_feats['vix'] = vix_aligned
        
        # 4. VIX change rate (over 5 bars)
        df_feats['vix_change'] = vix_aligned.pct_change(periods=5)
        
        # 5. VIX-SPY rolling correlation (20 periods)
        # Negative correlation is normal; if it breaks, something unusual is happening
        spy_ret = prices.pct_change()
        vix_ret = vix_aligned.pct_change()
        df_feats['vix_spy_corr'] = spy_ret.rolling(window=20).corr(vix_ret)
    
    return df_feats.dropna()

def train_meta_model(X, y, model_type='mlp'):
    """
    Trains a Meta-Model.
    Args:
        X: features
        y: target labels
        model_type: 'rf' for RandomForestClassifier, 'mlp' for PyTorchMetaLabelClassifier
    """
    # Align indices
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    if len(X) < 50:
        print("Not enough data to train.")
        return None
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # Train
    if model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced_subsample', random_state=42)
    elif model_type == 'mlp':
        from src.dl.models import PyTorchMetaLabelClassifier
        # Hyperparameters for MLP
        clf = PyTorchMetaLabelClassifier(
            hidden_dim=64,
            dropout=0.3,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=40,
            batch_size=64,
            patience=10
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    print(f"--- Meta-Model Performance ({model_type.upper()}) ---")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf

if __name__ == "__main__":
    # Test Run with Fake Data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
    price = 100 + np.cumsum(np.random.normal(0, 1, 1000))
    prices = pd.Series(price, index=dates)
    
    print("Generating Features...")
    vol = get_daily_vol(prices)
    X = get_features(prices, vol)
    
    print("Generating Labels...")
    primary_side = primary_signal(prices)
    # Filter only where we have a signal
    t_events = primary_side[primary_side != 0].index
    
    # Align t_events with volatility index
    t_events = t_events.intersection(vol.index)
    
    vertical_barrier = pd.Series(t_events + pd.Timedelta(days=10), index=t_events)
    
    # Triple Barrier
    pt_sl = [1, 1] # 1x Volatility for PT and SL
    min_ret = 0.005 # Min 0.5% target
    
    events = get_events(prices, t_events, pt_sl, vol, min_ret, vertical_barrier_times=vertical_barrier, side=primary_side)
    labels = get_bins(events, prices)
    
    print(f"Labeled Events: {len(labels)}")
    
    print("Training Model...")
    train_meta_model(X, labels['bin'])

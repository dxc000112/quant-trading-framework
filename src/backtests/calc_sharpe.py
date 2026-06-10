import pandas as pd
import numpy as np
import joblib
from src.data_loader import fetch_data
from src.labeling import get_daily_vol
from src.meta_model import get_features

def calculate_sharpe():
    print("--- 📉 Calculating Sharpe Ratio for Reversal Strategy ---")
    
    # 1. Load Model
    try:
        model_data = joblib.load('rf_model_short.pkl')
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        print("✅ Model Loaded.")
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return

    # 2. Fetch Data (1 year as a solid sample)
    print("Fetching 1 year of 5-min block data for backtesting...")
    df = fetch_data("SPY", period="365d", interval="5m")
    
    if df is None or df.empty:
        print("No data.")
        return

    # 3. Features
    print("Calculating Features & VWAP...")
    vol = get_daily_vol(df['Close'])
    X = get_features(df, vol)
    
    # Align
    df = df.loc[X.index].copy()
    X = X.fillna(0)
    
    # Predict
    probs = model.predict_proba(X)
    df['conf'] = probs[:, 1]
    
    # Setup for daily tracking
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df['local_time'] = df.index.tz_convert('America/New_York')
    df['date'] = df['local_time'].dt.date
    df['day_of_week'] = df['local_time'].dt.day_name()
    
    # Calculate VWAP
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['vol_x_price'] = df['Volume'] * df['typical_price']
    
    daily_vwap = []
    
    for date in df['date'].unique():
        df_day = df[df['date'] == date].copy()
        df_day['cum_vol_price'] = df_day['vol_x_price'].cumsum()
        df_day['cum_vol'] = df_day['Volume'].cumsum()
        df_day['vwap_daily'] = df_day['cum_vol_price'] / df_day['cum_vol']
        daily_vwap.append(df_day['vwap_daily'])
        
    df['vwap'] = pd.concat(daily_vwap)
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    
    # Parameters matches Live Bot
    TP_PCT = 0.008 
    SL_PCT = 0.005 
    
    # We will log daily returns to calculate Sharpe
    daily_pnl = {}
    
    dates = sorted(df['date'].unique())
    print(f"Running simulation over {len(dates)} days...")
    
    total_trades = 0
    wins = 0

    import datetime
    t_930 = datetime.time(9, 30)
    t_1100 = datetime.time(11, 0)
    
    for date in dates:
        df_day = df[df['date'] == date]
        
        in_position = False
        entry_price = 0.0
        tp_price = 0.0
        sl_price = 0.0
        
        day_pnl = 0.0
        
        for i in range(len(df_day)):
            row = df_day.iloc[i]
            t = row['local_time'].time()
            conf = row['conf']
            price = row['Close']
            high = row['High']
            low = row['Low']
            
            if in_position:
                if low <= sl_price:
                    day_pnl += (sl_price - entry_price)
                    in_position = False
                elif high >= tp_price:
                    day_pnl += (tp_price - entry_price)
                    wins += 1
                    in_position = False
                elif conf >= 0.42:
                    day_pnl += (price - entry_price)
                    if price > entry_price: wins += 1
                    in_position = False
                elif t >= datetime.time(15, 55):
                    day_pnl += (price - entry_price)
                    if price > entry_price: wins += 1
                    in_position = False
                continue
                
            is_open = (t >= t_930 and t < t_1100)
            
            # --- APPLY FILTERS ---
            # 1. Day of Week Filter: No Fridays
            is_valid_day = (row['day_of_week'] != 'Friday')
            
            # 2. Time Filter: Only 9:30 to 10:00 AM (More momentum)
            # t_1000 = datetime.time(10, 0)
            # is_open = (t >= t_930 and t < t_1000)
            
            # 3. Trend Filter: Price MUST be > VWAP AND > 50-SMA
            # We want to buy pullbacks in an UPTREND, not catch falling knives in a DOWNTREND.
            is_uptrend = (price > row['vwap']) and (price > row['sma_50'])
            
            if is_open and conf < 0.20 and not in_position and is_valid_day and is_uptrend:
                entry_price = price
                tp_price = entry_price * (1 + TP_PCT)
                sl_price = entry_price * (1 - SL_PCT)
                in_position = True
                total_trades += 1
                
        # Record this day's net PnL 
        # Convert to percentage return based on an assumed account size or base SPY price?
        # A simpler approach: Return % = Day PnL / SPY Price
        if len(df_day) > 0:
            start_price = df_day['Open'].iloc[0]
            daily_pct_return = day_pnl / start_price
            daily_pnl[date] = daily_pct_return

    # --- Metrics Calc ---
    returns_series = pd.Series(daily_pnl)
    
    if len(returns_series) < 2 or total_trades == 0:
        print("Not enough trades/days to calculate meaningful Sharpe.")
        return
        
    avg_daily_return = returns_series.mean()
    std_daily_return = returns_series.std()
    
    # Assume Risk-Free Rate = 4.5% annualized -> 0.045 / 252 daily
    rf_daily = 0.045 / 252
    
    # Calculate Sharpe
    if std_daily_return == 0:
        sharpe_ratio = 0.0
    else:
        # Annualized Sharpe Ratio = (Mean Return - RF) / Std Dev * sqrt(252)
        sharpe_ratio = ((avg_daily_return - rf_daily) / std_daily_return) * np.sqrt(252)
        
    print("\n" + "="*40)
    print("🏆 PERFORMANCE METRICS (Past 365 Days)")
    print("="*40)
    print(f"Total Trading Days: {len(dates)}")
    print(f"Total Trades Taken: {total_trades}")
    print(f"Win Rate: {wins/total_trades*100:.1f}%" if total_trades > 0 else "Win Rate: N/A")
    print(f"Avg Daily Return:   {avg_daily_return*100:.3f}%")
    print(f"Daily Volatility:   {std_daily_return*100:.3f}%")
    
    # Max Drawdown
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    print(f"Max Drawdown:       {max_dd*100:.2f}%")
    
    print("-" * 40)
    if sharpe_ratio < 0:
        eval_str = "🔴 Poor (Negative Return)"
    elif sharpe_ratio < 1:
        eval_str = "🟡 Sub-par (Risk > Reward)"
    elif sharpe_ratio < 2:
        eval_str = "🟢 Good"
    else:
        eval_str = "🚀 Excellent"
        
    print(f"📈 Annualized SHARPE RATIO: {sharpe_ratio:.2f}  |  {eval_str}")
    print("="*40)
    
if __name__ == "__main__":
    calculate_sharpe()

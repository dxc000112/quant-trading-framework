import pandas as pd
import numpy as np
from src.data_loader import fetch_data
from src.strategy_trend import check_trend_signal
import datetime

import joblib
from src.labeling import get_daily_vol
from src.meta_model import get_features

def replay_trend_strategy(date_str=None):
    """
    Replays the VWAP Trend Strategy for a specific day.
    """

    print(f"--- Replaying VWAP Trend Strategy (1 Year) ---")

    
    # Load ML Model
    try:
        model_data = joblib.load('rf_model_short.pkl')
        print(f"✅ Loaded: {type(model_data)}")
        if isinstance(model_data, dict):
             # Try to find the model inside the dict
             if 'model' in model_data:
                 model = model_data['model']
                 print("   -> Extracted 'model' key.")
             else:
                 print("   -> ❌ Dict loaded but no 'model' key found.")
                 model = None
        else:
             model = model_data
             
    except Exception as e:
        print(f"❌ Model load error: {e}")
        model = None

    
    # Optimization: If filtering for a specific date, fetch less data
    if date_str:
        print(f"Optimization: Fetching 60 days for target {date_str}...")
        df = fetch_data("SPY", period="60d", interval="5m")
    else:
        print("Fetching 3 Years of 5m Data...")
        df = fetch_data("SPY", period="1000d", interval="5m")
    
    if df is None or df.empty:
        print("No data fetched.")
        return

    # Filter by Specific Date if provided
    if date_str:
        print(f"Filter active: Running only for {date_str}")
    
    print(f"Fetched {len(df)} bars. Starting Simulation...")
    run_simulation(df, model, target_date=date_str)


def run_simulation(df_full, model=None):
    # Group by Day first
    days = df_full.groupby(df_full.index.date)
    
    monthly_stats = {} # "2024-01-01": pnl
    yearly_stats = {} # "2024": pnl
    
    total_pnl = 0
    total_trades = 0
    wins = 0
    
    # ... (Indicators Calculation is same, skipping re-paste to save token unless needed) ...
    # Wait, the replace tool replaces the block. I need to keep the indicator logic or ensuring I don't delete it.
    # The target block covers lines 27 to 47 (Fetch + Filter + Start).
    # And then lines 49-58 (run_simulation header).
    # I should be careful. I will split this into two edits to be safe.
    # First edit: Update fetch logic and remove filter.





def run_simulation(df_full, model=None, target_date=None):
    # Group by Day first
    days = df_full.groupby(df_full.index.date)
    
    monthly_stats = {} # "2024-01-01": pnl
    total_pnl = 0
    total_trades = 0
    wins = 0
    
    for date, df_day in days:
        # Skip incomplete days or small data
        if len(df_day) < 30: continue
            
        has_traded_today = False
        timestamps = df_day.index
        
        # Optimization: Pre-calculate indicators for the whole day vectorized
        # This is strictly faster than windowing inside loop.
        # Check strategy logic: VWAP (intraday), EMA9, EMA21.
        
        # 1. VWAP Intraday
        df_day = df_day.copy()
        df_day['tp'] = (df_day['High'] + df_day['Low'] + df_day['Close']) / 3
        df_day['vp'] = df_day['tp'] * df_day['Volume']
        df_day['vwap'] = df_day['vp'].cumsum() / df_day['Volume'].cumsum()
        
        # 2. EMA Intraday (Reset every day? Or continuous?)
        # Strategy says: "EMA 9/21".
        # If we use 1m 5d chart, EMA carries over.
        # If we use 1d slices, EMA resets.
        # Real-time bot fetches "5d", so EMA carries over. 
        # But here we slice by day.
        # Better approximation: Calculate EMA on the whole DF_FULL before slicing days!
        pass
    
    # Correct Approach: Calculate Indicators on Full DF first
    print("Calculating Indicators...")
    df_full['tp'] = (df_full['High'] + df_full['Low'] + df_full['Close']) / 3
    df_full['vp'] = df_full['tp'] * df_full['Volume']
    
    # Group by date for VWAP reset (Standard Intraday VWAP)
    # This is heavy for loop. 
    # Vectorized group cumsum:
    df_full['date_group'] = df_full.index.date
    df_full['cum_vp'] = df_full.groupby('date_group')['vp'].cumsum()
    df_full['cum_vol'] = df_full.groupby('date_group')['Volume'].cumsum()
    df_full['vwap'] = df_full['cum_vp'] / df_full['cum_vol']
    
    # EMAs (Continuous)
    df_full['ema_9'] = df_full['Close'].ewm(span=9, adjust=False).mean()
    df_full['ema_21'] = df_full['Close'].ewm(span=21, adjust=False).mean()


    
    # ADX Calculation (14)
    alpha = 1/14
    df_full['tr0'] = df_full['High'] - df_full['Low']
    df_full['tr1'] = (df_full['High'] - df_full['Close'].shift(1)).abs()
    df_full['tr2'] = (df_full['Low'] - df_full['Close'].shift(1)).abs()
    df_full['tr'] = df_full[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    df_full['p_dm'] = df_full['High'] - df_full['High'].shift(1)
    df_full['n_dm'] = df_full['Low'].shift(1) - df_full['Low']
    df_full['p_dm'] = np.where((df_full['p_dm'] > df_full['n_dm']) & (df_full['p_dm'] > 0), df_full['p_dm'], 0.0)
    df_full['n_dm'] = np.where((df_full['n_dm'] > df_full['p_dm']) & (df_full['n_dm'] > 0), df_full['n_dm'], 0.0)
    
    df_full['tr_s'] = df_full['tr'].ewm(alpha=alpha, adjust=False).mean()
    df_full['p_dm_s'] = df_full['p_dm'].ewm(alpha=alpha, adjust=False).mean()
    df_full['n_dm_s'] = df_full['n_dm'].ewm(alpha=alpha, adjust=False).mean()
    
    df_full['p_di'] = 100 * (df_full['p_dm_s'] / df_full['tr_s'])
    df_full['n_di'] = 100 * (df_full['n_dm_s'] / df_full['tr_s'])
    
    df_full['dx'] = 100 * (abs(df_full['p_di'] - df_full['n_di']) / (df_full['p_di'] + df_full['n_di']))
    df_full['adx'] = df_full['dx'].ewm(alpha=alpha, adjust=False).mean()

    # --- AI/ML Feature Calculation (Vectorized) ---
    if model is not None:
        print("Calculating AI Confidence Scores...")
        try:
            # 1. Get Volatility
            vol = get_daily_vol(df_full['Close'])
            
            # 2. Get Features
            X = get_features(df_full, vol)
            
            # 3. Predict Proba (Vectorized)
            # This might be memory intensive for 1 year of 5m data (~20k rows), but should be fine.
            # Handle NaN features (first 50 rows)
            X = X.fillna(0) 
            
            # Predict
            probs = model.predict_proba(X)
            # Class 1 is 'Profitable Short'
            df_full['conf'] = probs[:, 1]
            
            # 4. Calculate Delta (Previous - Current)
            # Shift 1 is Previous
            df_full['conf_delta'] = df_full['conf'].shift(1) - df_full['conf']
            
        except Exception as e:
            print(f"Error calculating ML features: {e}")
            df_full['conf_delta'] = 0.0
    else:
        df_full['conf_delta'] = 0.0

    
    # Iterate Days again
    days = df_full.groupby('date_group')
    
    PULLBACK_THRESHOLD = 0.005
    
    print("Simulating Days...")
    for date, df_day in days:
        if target_date and str(date) != target_date:
            continue

        if len(df_day) < 10: continue
        
        day_pnl = 0
        month_key = date.strftime("%Y-%m")
        if month_key not in monthly_stats: monthly_stats[month_key] = 0
        
        # Loop bars
        # 5min bars. 9:30 is index 0. 
        # 10:00 is around index 6. 
        # Let's start scanning from index 3 (9:45).
        
        match_found = False
        
        for i in range(3, len(df_day)):
            row = df_day.iloc[i]
            price = row['Close']
            vwap = row['vwap']
            ema9 = row['ema_9']
            ema21 = row['ema_21']
            adx_val = row['adx'] 
            
            # ADX Filter removed as per user request (returning to raw trend)
            # if adx_val < 25:
            #    continue


            
            # Long
            if price > vwap and ema9 > ema21:
                dist = (price - ema9) / ema9

                if dist < PULLBACK_THRESHOLD and price > ema21:
                    
                    # --- AI Filter (Long Guard) ---
                    # If Short Conf > 0.7, assume Top, Skip Long.
                    skip_long = False
                    if 'conf' in row:
                        if row['conf'] > 0.7:
                            # print(f"   ⚠️ AI Filtered Long (Conf: {row['conf']:.2f})")
                            skip_long = True
                            
                    if not skip_long:
                        # ENTRY (LONG)
                        entry_price = price
                        signal = 1
                        
                        print(f"[{row.name}] 🚀 LONG ENTRY @ {entry_price:.2f}")
                        
                        tp_price = entry_price * 1.008
                        sl_price = entry_price * 0.995
                        
                        # Scan future bars for exit

                        exit_price = df_day['Close'].iloc[-1] # Default EOD
                        exit_reason = "EOD"
                        
                        for j in range(i+1, len(df_day)):
                            bar = df_day.iloc[j]
                            
                            # Check SL first (Conservative)
                            if bar['Low'] <= sl_price:
                                exit_price = sl_price
                                exit_reason = "SL"
                                break
                            if bar['High'] >= tp_price:
                                exit_price = tp_price
                                exit_reason = "TP"
                                break
                        
                        # Calculate PnL (Long: Exit - Entry)
                        pnl = exit_price - entry_price
                            
                        print(f"     -> Result: {exit_reason} @ {exit_price:.2f}. PnL: {pnl:.2f}")
                        
                        if pnl > 0: wins += 1
                        total_trades += 1
                        day_pnl += pnl
                        has_traded_today = True
                        break # Done for the day

                    
        monthly_stats[month_key] += day_pnl
        total_pnl += day_pnl
        
    # Report
    # Report
    print("\n--- 📆 Monthly Performance ---")
    yearly_stats = {}
    
    for m, p in monthly_stats.items():
        print(f"{m}: ${p:6.2f}")
        year = m.split("-")[0]
        if year not in yearly_stats: yearly_stats[year] = 0
        yearly_stats[year] += p
        
    print("\n--- 📅 Yearly Performance (Annualized) ---")
    for y, p in yearly_stats.items():
        print(f"{y}: ${p:6.2f}")
        
    print("\n--- 📊 Overall Stats ---")
    print(f"Total PnL: ${total_pnl:.2f}")
    if len(yearly_stats) > 0:
        avg_annual = total_pnl / len(yearly_stats)
        print(f"Avg Annual PnL: ${avg_annual:.2f}")
    
    print(f"Total Trades: {total_trades}")
    if total_trades > 0:
        print(f"Win Rate: {wins/total_trades:.1%}")
    else:
        print("Win Rate: N/A")


if __name__ == "__main__":
    replay_trend_strategy()

if __name__ == "__main__":
    replay_trend_strategy()

import pandas as pd
import numpy as np
import joblib
from src.data_loader import fetch_data
from src.labeling import get_daily_vol
from src.meta_model import get_features

def replay_reversal_strategy(target_date_str=None):
    print("--- 🦅 Replaying Reversal Sniper Strategy (Live Logic) ---")
    
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

    # 2. Fetch Data
    # To replicate "Live" exactly, we must limit the history to what the bot saw (e.g. 5 days)
    # The Live Bot uses period="5d". 
    # If target is 2026-01-20, "5d" means Jan 15 to Jan 20.
    
    print("Fetching Data (Buffer)...")
    # Fetch enough data first
    df = fetch_data("SPY", period="20d", interval="5m")
    
    if df is None or df.empty:
        print("No data.")
        return

    # Simulate the "Live Window" for the target date
    if target_date_str:
        if df.index.tz is None:
             df.index = df.index.tz_localize('UTC')
        df = df.tz_convert('America/New_York')
        
        target_dt = pd.Timestamp(target_date_str).tz_localize('America/New_York')
        
        # Live bot logic: start_dt = now - 5 days
        # So for Jan 20, start = Jan 15.
        start_dt = target_dt - pd.Timedelta(days=5)
        
        # Slice the dataframe to this window [Start, Target End]
        # We need data UP TO the end of the target date.
        end_dt = target_dt + pd.Timedelta(days=1) # Include full target day
        
        df_live_view = df[(df.index >= start_dt) & (df.index < end_dt)].copy()
        
        print(f"Simulating Live Window: {start_dt.date()} to {target_dt.date()}")
        print(f"Bars in Window: {len(df_live_view)}")
        
        if df_live_view.empty:
            print("No data in window.")
            return
            
        # Use this slice for feature calc
        df = df_live_view
        
    else:
        # Default behavior if no date
        pass

    # 3. Calculate Features (Vectorized) on the mimic window
    print("Calculating Features & Confidence on Restricted Window...")
    try:
        vol = get_daily_vol(df['Close'])
        X = get_features(df, vol)
        
        # FIX: Align df with X (get_features drops NaNs)
        df = df.loc[X.index].copy()
        
        # Handle NaNs
        X = X.fillna(0)
        
        # Predict
        probs = model.predict_proba(X)
        df['conf'] = probs[:, 1] # Short Confidence
        
    except Exception as e:
        print(f"Feature Calc Error: {e}")
        return

    # Filter Again for the specific target day simulation loop
    # We computed features on [Jan 15 - Jan 20].
    # Now we only want to simulate trades for Jan 20.
    if target_date_str:
        target_dt_date = pd.Timestamp(target_date_str).date()
        df = df[df.index.date == target_dt_date]
        if df.empty:
            print(f"No data left for target date {target_date_str} after processing.")
            return

    # 4. Simulate
    print("Simulating Trades...")
    
    total_trades = 0
    wins = 0
    total_pnl = 0.0
    
    # Parameters matches Live Bot:
    # TP: 1.008 (+0.8%)
    # SL: 0.995 (-0.5%)
    TP_PCT = 0.008 
    SL_PCT = 0.005 
    
    trade_log = []
    
    in_position = False
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    
    bars = df.itertuples()
    
    # Time Windows
    import datetime
    processed_minutes = set()

    for row in bars:
        price = row.Close
        conf = getattr(row, 'conf', 0.0)
        
        # Check Exit if in position
        if in_position:
            # Check Low for SL, High for TP
            if row.Low <= sl_price:
                pnl = sl_price - entry_price
                total_pnl += pnl
                in_position = False
                trade_log.append(f"🔴 SL  | {row.Index.strftime('%H:%M')} | {pnl:.2f}")
            elif row.High >= tp_price:
                pnl = tp_price - entry_price
                total_pnl += pnl
                wins += 1
                in_position = False
                trade_log.append(f"🟢 TP  | {row.Index.strftime('%H:%M')} | {pnl:.2f}")
            
            # Dynamic Exit: Conf >= 0.42
            elif conf >= 0.42:
                pnl = price - entry_price # Exit at Close of bar
                total_pnl += pnl
                if pnl > 0: wins += 1
                in_position = False
                trade_log.append(f"⚡ DYN | {row.Index.strftime('%H:%M')} | {pnl:.2f} (Conf {conf:.2f})")

            elif row.Index.hour >= 15 and row.Index.minute >= 55:
                # EOD Exit
                pnl = price - entry_price
                total_pnl += pnl
                if pnl > 0: wins += 1
                in_position = False
                trade_log.append(f"⚪ EOD | {row.Index.strftime('%H:%M')} | {pnl:.2f}")
                
            continue
            
        # Entry Logic (Live Bot Style)
        # 1. Check Time Window (Morning Only as per user request)
        t = row.Index.time()
        is_open = (t >= datetime.time(9, 30) and t < datetime.time(11, 0))
        # is_close removed
        
        if is_open:
            # 2. Check Confidence Threshold (< 0.20)
            if conf < 0.20:
                print(f"⚠️ Low Conf Detected: {row.Index.strftime('%H:%M')} | {conf:.2f} | In Position: {in_position}")
                
                # 3. Prevent Duplicate Entries
                if in_position:
                    continue

                # TRIGGER LONG (Reversal)
                entry_price = price
                tp_price = entry_price * (1 + TP_PCT)
                sl_price = entry_price * (1 - SL_PCT)
                
                in_position = True
                total_trades += 1
                
                print(f"🚀 ENTRY: {row.Index.strftime('%H:%M')} @ {entry_price:.2f} (Conf: {conf:.2f})")

    # Stats Analysis
    if not df.empty:
        # 1. Verify Price Drop
        open_price = df['Open'].iloc[0]
        close_price = df['Close'].iloc[-1]
        pct_change = (close_price - open_price) / open_price
        print(f"\n📉 Price Action: Open {open_price:.2f} -> Close {close_price:.2f} ({pct_change:.2%})")
        
        # 2. Extract Top 5 drops (largest 5m red candles) and their confidence
        df['candle_ret'] = (df['Close'] - df['Open']) / df['Open']
        worst_candles = df.sort_values('candle_ret').head(5)
        print("\n💥 Top 5 Bearish Candles vs Confidence:")
        print(worst_candles[['Close', 'candle_ret', 'conf']].to_string())
        
        # INVESTIGATE 10:02
        print("\n🔍 Investigating 10:02 Candle:")
        # Locate rows between 10:00 and 10:05
        mask_time = df.index.indexer_between_time('10:00', '10:05')
        print(df.iloc[mask_time][['Close', 'conf']].to_string())


        # 3. Inspect Features when Confidence was LOW (e.g. 10:00 - 14:00)
        # Why was it ~0.22?
        # Let's pick a representative row near the mode (0.22)
        mode_row = df[ df['conf'].between(0.21, 0.23) ].iloc[0]
        print(f"\n🧐 Inspecting Typical Low Conf Row ({mode_row.name.strftime('%H:%M')}):")
        print(f"Conf: {mode_row['conf']:.2f}")
        # Print first 10 features
        # We need to map 'X' columns back if possible, but they are not stored in df directly except via X
        # We can re-merge or just assume they are available if we concatenated. 
        # Actually replay script discards X.
        
    print("\nNote: To see specific features, we need to inspect the 'X' dataframe.")




    # Report
    print(f"\n--- 📊 Results {target_date_str if target_date_str else ''} ---")
    print(f"Total Trades: {total_trades}")
    if total_trades > 0:
        print(f"Win Rate: {wins/total_trades:.1%}")
        print(f"Total PnL (per share): ${total_pnl:.2f}")
        
    else:
        print("No trades triggered.")
        
    print("\n--- Trade Log ---")
    for t in trade_log:
        print(t)

if __name__ == "__main__":
    import sys
    # Default to yesterday if no arg
    if len(sys.argv) > 1:
        date_arg = sys.argv[1]
    else:
        date_arg = "2026-01-20"
        
    replay_reversal_strategy(date_arg)

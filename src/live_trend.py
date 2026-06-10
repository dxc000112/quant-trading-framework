import time
import os
import pandas as pd
import datetime
from dotenv import load_dotenv
import logging
from src.data_loader import fetch_data
from src.strategy_trend import check_trend_signal
from src.notifier import DiscordNotifier

import joblib
from src.labeling import get_daily_vol
from src.meta_model import get_features

# Setup
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
TICKER = "SPY"
INTERVAL = "5m"
PERIOD = "5d" # Need today's data for VWAP
MODEL_PATH = 'rf_model_short.pkl' # Path to ML Model

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("trend_bot.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info(f"--- Live Trend Bot ({TICKER}) Started ---")
    
    notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
    
    # Load ML Model
    try:
        model_data = joblib.load(MODEL_PATH)
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            logging.info(f"✅ ML Model Loaded (extracted from dict): {MODEL_PATH}")
        else:
            model = model_data
            logging.info(f"✅ ML Model Loaded: {MODEL_PATH}")
            
    except Exception as e:
        logging.error(f"❌ Failed to load ML Model: {e}")
        model = None

        
    prev_conf = None # Track previous confidence for Delta check
    
    # State tracking
    last_trade_date = None
    in_position = False
    
    # Simple cooldown to avoid spamming the same signal minute after minute
    # But ideally we want to take the FIRST one.
    
    while True:
        try:
            now = datetime.datetime.now()
            current_date = now.date()
            
            # --- 1. Time Check ---
            # User wants "Morning Market" logic. 
            # Ideally 9:45 AM - 11:30 AM ET roughly.
            # We can rely on visual supervision or logic. 
            # For now, let's just run continuously, but maybe skip pre-market if data is messy.
            
            # logging.info(f"Scanning Trend...", end=" ", flush=True) # logging adds newline automatically
            
            # --- 2. Fetch Data ---
            df = fetch_data(TICKER, PERIOD, INTERVAL)
            
            if df is None or df.empty:
                logging.warning("Error fetching data.")
                time.sleep(60)
                continue
                
            curr_price = df['Close'].iloc[-1]
            
            current_conf = None # Reset for this loop
            
            # --- 3. ML Confidence Check (Support Monitor) ---
            if model is not None:

                try:
                    # Calculate Features
                    # Note: get_features expects 'Close', etc.
                    vol = get_daily_vol(df['Close'])
                    X = get_features(df, vol)
                    
                    # Predict Confidence
                    # [0][1] is probability of Class 1 (Short Success)
                    # We usually want Class 1 as the bearish signal.
                    # Wait, train_short.py labels 1 as "Profitable Short".
                    current_conf = model.predict_proba(X.iloc[[-1]])[0][1]
                    
                    # logging.info(f"Conf: {current_conf:.2f}") 
                    
                    # Delta Check (Collapse)
                    if prev_conf is not None:
                        delta = prev_conf - current_conf
                        if delta > 0.05:
                            # TRIGGER: Confidence Collapse
                            msg = (f"📉 **Confidence Collapse (Support Found)**\n"
                                   f"Ticker: {TICKER}\n"
                                   f"Conf Drop: {prev_conf:.2f} -> {current_conf:.2f}\n"
                                   f"Delta: {delta:.2f} (> 0.05)\n"
                                   f"Price: {curr_price:.2f}\n"
                                   f"Action: **COVER SHORT / WAIT**")
                            notifier.send_alert(f"ML Alert: Support {TICKER}", msg, 0xffff00) # Yellow
                            logging.warning(f"Confidence Collapse Detected: {prev_conf:.2f} -> {current_conf:.2f}")
                            
                    prev_conf = current_conf
                    
                except Exception as e:
                    logging.error(f"(ML Error: {e})")
            
            log_msg = f"Price: {curr_price:.2f}"
            if current_conf is not None:
                log_msg = f"Conf: {current_conf:.2f} " + log_msg
            
            logging.info(f"Scanning... {log_msg}")


            # --- 4. Trend Strategy Check ---
            signal = check_trend_signal(df)
            
            # Only check if we haven't traded today (Limit 1 per day for Small Account)
            if last_trade_date != current_date:
                
                curr_price = df['Close'].iloc[-1]
                
                if signal == 1:
                    # --- AI Filter (Long Guard) ---
                    # If AI Model Confidence (Short Probability) is HIGH (> 0.7), skip Long.
                    skip_long = False
                    if current_conf is not None and current_conf > 0.7:
                         logging.warning(f"⚠️ Top Detected by AI (Conf: {current_conf:.2f}). Skipping Long.")
                         notifier.send_alert(f"⚠️ Filtered Long: {TICKER}", 
                                             f"Trend wants to Buy, but AI predicts Top.\nShort Conf: {current_conf:.2f} (> 0.7)", 
                                             0xffaa00)
                         skip_long = True
                    
                    if not skip_long:
                        logging.info(" -> LONG SIGNAL! (Call)")
                        # Calculate TP/SL
                        tp_price = curr_price * 1.008
                        sl_price = curr_price * 0.995
                        
                        msg = (f"🚀 **LONG ENTRY (Trend Pullback)**\n"
                               f"Ticker: {TICKER}\n"
                               f"Entry Price: {curr_price:.2f}\n"
                               f"🎯 **Target (TP)**: {tp_price:.2f} (+0.8%)\n"
                               f"🛑 **Stop (SL)**: {sl_price:.2f} (-0.5%)\n"
                               f"Logic: Price > VWAP, Pullback to EMA9")
                        
                        notifier.send_alert(f"Trend Buy: {TICKER}", msg, 0x00ff00)
                        
                        last_trade_date = current_date
                        in_position = True # Conceptual
                    
                # elif signal == -1: ... (Short Disabled in Strategy)

                    
                else:
                    # logging.info(f"(No Signal) Price: {curr_price:.2f}")
                    pass
            else:
                logging.info(f"(Daily Limit Reached - Done for {current_date})")
                
            time.sleep(60)

        except KeyboardInterrupt:
            logging.info("\nStopping...")
            break
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()


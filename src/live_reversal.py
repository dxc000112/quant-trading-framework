import time
import os
import pandas as pd
import datetime
import pytz
from dotenv import load_dotenv
import logging
from src.data_loader import fetch_data
from src.notifier import DiscordNotifier

import joblib
from src.labeling import get_daily_vol
from src.meta_model import get_features

# Setup
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
TICKER = "SPY"
INTERVAL = "5m"
PERIOD = "5d" 
MODEL_PATH = 'rf_model_short.pkl'

# Configure Logging (Separate log file for Reversal Bot)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("reversal_bot.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info(f"--- 🦅 Reversal Sniper Bot ({TICKER}) Started ---")
    
    notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
    
    # Load ML Model
    try:
        model_data = joblib.load(MODEL_PATH)
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            logging.info(f"✅ ML Model Loaded: {MODEL_PATH}")
        else:
            model = model_data
            logging.info(f"✅ ML Model Loaded: {MODEL_PATH}")
            
    except Exception as e:
        logging.error(f"❌ Failed to load ML Model: {e}")
        return

    # Tracking State
    prev_conf = None 
    
    while True:
        try:
            # 1. Fetch Data
            df = fetch_data(TICKER, PERIOD, INTERVAL)
            
            if df is None or df.empty:
                logging.warning("Error fetching data. Retrying in 60s...")
                time.sleep(60)
                continue
                
            curr_price = df['Close'].iloc[-1]
            
            # --- CALCULATE VWAP & SMA50 ---
            df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['vol_x_price'] = df['Volume'] * df['typical_price']
            
            # Calculate daily VWAP (ensure timezone is consistent with fetch_data)
            # Assuming fetch_data returns simple df where we just cumsum the fetched window
            # A more robust live VWAP calculation needs daily resets, but for 5d window:
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df['local_time'] = df.index.tz_convert('America/New_York')
            df['date'] = df['local_time'].dt.date
            
            daily_vwap = []
            for date in df['date'].unique():
                df_day = df[df['date'] == date].copy()
                df_day['cum_vol_price'] = df_day['vol_x_price'].cumsum()
                df_day['cum_vol'] = df_day['Volume'].cumsum()
                df_day['vwap_daily'] = df_day['cum_vol_price'] / df_day['cum_vol']
                daily_vwap.append(df_day['vwap_daily'])
                
            df['vwap'] = pd.concat(daily_vwap)
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            
            curr_vwap = df['vwap'].iloc[-1]
            curr_sma50 = df['sma_50'].iloc[-1]
            
            # 2. Calculate AI Confidence
            try:
                vol = get_daily_vol(df['Close'])
                X = get_features(df, vol)
                
                # Predict (Class 1 = Profitable Short)
                # We are looking for this to PEAK (Bearish) and then COLLAPSE (Bullish Signal)
                current_conf = model.predict_proba(X.iloc[[-1]])[0][1]
                
            except Exception as e:
                logging.error(f"ML Feature Error: {e}")
                time.sleep(60)
                continue

            # 3. Logic: Low Confidence Reversal (Bull Put Spread)
            # Strategy: If Conf < 0.20 during Open/Close windows, Sell Put Spread.
            
            log_msg = f"Price: {curr_price:.2f} | Conf: {current_conf:.2f}"
            
            # Time Windows (ET): 9:30-11:00 AND 14:30-16:00
            # FIX: Convert local system time to NY Time
            tz_ny = pytz.timezone('America/New_York')
            now_ny = datetime.datetime.now(tz_ny)
            t = now_ny.time()
            
            is_open = (t >= datetime.time(9, 30) and t < datetime.time(11, 0))
            is_window = is_open
            
            # --- APPLY FILTERS ---
            day_of_week = now_ny.strftime('%A')
            is_valid_day = (day_of_week != 'Friday')
            is_uptrend = (curr_price > curr_vwap) and (curr_price > curr_sma50)
            
            # 1. Entry Logic
            if is_window and current_conf < 0.20 and is_valid_day and is_uptrend:
                
                # --- OPTION LOGIC: Bull Put Spread ---
                # Sell ATM Put / Buy ATM-1 Put
                strike_short = int(curr_price) # Round down to nearest integer
                strike_long = strike_short - 1
                
                logging.info(f"🚀 LOW CONF SIGNAL TRIGGERED!")
                logging.info(f"   Conf: {current_conf:.2f} (< 0.20)")
                
                # Calculate Targets (Share Logic)
                tp_price = curr_price * 1.008 
                sl_price = curr_price * 0.995 
                
                msg = (f"🦅 **REVERSAL SNIPER: {TICKER}**\n"
                       f"🔥 **LONG ENTRY (Low Conf)**\n"
                       f"Price: {curr_price:.2f}\n"
                       f"📉 Conf: **{current_conf:.2f}** (< 0.20)\n"
                       f"⏰ Time: {t.strftime('%H:%M')}\n\n"
                       f"🛡️ **OPTION PLAY: Bull Put Spread (Recommend)**\n"
                       f"**SELL**: ${strike_short} PUT\n"
                       f"**BUY**:  ${strike_long} PUT\n"
                       f"Credit: ~$0.40 | Margin: $100\n"
                       f"Logic: Eat Time Decay (Theta). Profit if Flat or Up.")
                
                notifier.send_alert(f"🦅 Reversal Signal: {TICKER}", msg, 0x00ffff) # Cyan Color
                
                # Cooldown to prevent spamming same minute
                # time.sleep(300) # Cooldown Disabled 
            
            # 2. Dynamic Exit Logic (Conf >= 0.41)
            # Only alert if we just crossed the threshold (to avoid spamming every minute)
            elif (prev_conf is None or prev_conf < 0.41) and current_conf >= 0.41:
                logging.info(f"⚡ DYNAMIC EXIT SIGNAL TRIGGERED!")
                logging.info(f"   Conf: {current_conf:.2f} (>= 0.41)")
                
                msg = (f"🦅 **REVERSAL SNIPER: {TICKER}**\n"
                       f"⚡ **DYNAMIC TAKE PROFIT ALERT**\n"
                       f"Price: {curr_price:.2f}\n"
                       f"📈 Conf: **{current_conf:.2f}** (Rising Risk)\n"
                       f"⏰ Time: {t.strftime('%H:%M')}\n\n"
                       f"💡 **Action**: Close Long / Take Profit\n"
                       f"Reason: Model confidence rising indicates bearish pressure returning.")
                
                notifier.send_alert(f"⚡ Exit Alert: {TICKER}", msg, 0xff00ff) # Magenta Color

            else:
                pass
            
            logging.info(f"Scanning... {log_msg}")
            
            # Update State
            prev_conf = current_conf
            
            # Wait
            time.sleep(60)

        except KeyboardInterrupt:
            logging.info("Stopping...")
            break
        except Exception as e:
            logging.error(f"Loop Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()

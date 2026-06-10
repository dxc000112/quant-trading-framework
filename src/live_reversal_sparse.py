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
INTERVAL = "15m"       # 15-minute bars (coarser granularity)
PERIOD = "10d"          # 10-day window (more context)
MODEL_PATH = 'rf_model_short.pkl'
SCAN_INTERVAL = 300     # 5 minutes between scans

# Thresholds
LONG_ENTRY_CONF = 0.20   # Conf < 0.20 = strong bullish (same as original)
SHORT_SIGNAL_CONF = 0.55 # Conf >= 0.55 = bearish signal (buy Put)
EXIT_WARNING_CONF = 0.41 # Conf >= 0.41 = exit warning (same as original)

# Configure Logging (Separate log file)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("reversal_sparse_bot.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info(f"--- 🦅 Sparse Reversal Bot ({TICKER}) Started ---")
    logging.info(f"   Interval: {INTERVAL} | Window: {PERIOD} | Scan: {SCAN_INTERVAL}s")
    logging.info(f"   Long Entry: Conf < {LONG_ENTRY_CONF}")
    logging.info(f"   Short Signal: Conf >= {SHORT_SIGNAL_CONF}")
    
    notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
    
    # Load ML Model
    try:
        model_data = joblib.load(MODEL_PATH)
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
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
            # 1. Fetch 15-minute bar data (10-day window)
            df = fetch_data(TICKER, PERIOD, INTERVAL)
            
            if df is None or df.empty:
                logging.warning("Error fetching data. Retrying in 60s...")
                time.sleep(60)
                continue
                
            curr_price = df['Close'].iloc[-1]
            
            # --- CALCULATE VWAP & SMA50 ---
            df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['vol_x_price'] = df['Volume'] * df['typical_price']
            
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
                
                current_conf = model.predict_proba(X.iloc[[-1]])[0][1]
                
            except Exception as e:
                logging.error(f"ML Feature Error: {e}")
                time.sleep(60)
                continue

            # 3. Time check
            tz_ny = pytz.timezone('America/New_York')
            now_ny = datetime.datetime.now(tz_ny)
            t = now_ny.time()
            
            is_open = (t >= datetime.time(9, 30) and t < datetime.time(11, 0))
            day_of_week = now_ny.strftime('%A')
            is_valid_day = (day_of_week != 'Friday')
            is_uptrend = (curr_price > curr_vwap) and (curr_price > curr_sma50)
            
            log_msg = f"Price: {curr_price:.2f} | Conf: {current_conf:.2f}"

            # ============================
            # SIGNAL LOGIC
            # ============================

            # A. SHORT SIGNAL: Conf >= 0.55 (Buy Put)
            if (prev_conf is None or prev_conf < SHORT_SIGNAL_CONF) and current_conf >= SHORT_SIGNAL_CONF:
                logging.info(f"🔴 SHORT SIGNAL TRIGGERED!")
                logging.info(f"   Conf: {current_conf:.2f} (>= {SHORT_SIGNAL_CONF})")
                
                strike = int(curr_price)
                
                msg = (f"🦅 **SPARSE REVERSAL: {TICKER}**\n"
                       f"🔴 **SHORT SIGNAL (High Conf)**\n"
                       f"Price: {curr_price:.2f}\n"
                       f"📈 Conf: **{current_conf:.2f}** (>= {SHORT_SIGNAL_CONF})\n"
                       f"⏰ Time: {t.strftime('%H:%M')}\n\n"
                       f"🐻 **OPTION PLAY: Buy Put**\n"
                       f"**BUY**: ${strike} PUT (ATM)\n"
                       f"Logic: Model sees high bearish probability on 15m timeframe.")
                
                notifier.send_alert(f"🔴 Sparse Short Signal: {TICKER}", msg, 0xff0000)

            # B. LONG ENTRY: Conf < 0.20 (during open window, same as original)
            elif is_open and current_conf < LONG_ENTRY_CONF and is_valid_day and is_uptrend:
                strike_short = int(curr_price)
                strike_long = strike_short - 1
                
                logging.info(f"🚀 LOW CONF SIGNAL TRIGGERED!")
                logging.info(f"   Conf: {current_conf:.2f} (< {LONG_ENTRY_CONF})")
                
                msg = (f"🦅 **SPARSE REVERSAL: {TICKER}**\n"
                       f"🔥 **LONG ENTRY (Low Conf)**\n"
                       f"Price: {curr_price:.2f}\n"
                       f"📉 Conf: **{current_conf:.2f}** (< {LONG_ENTRY_CONF})\n"
                       f"⏰ Time: {t.strftime('%H:%M')}\n\n"
                       f"🛡️ **OPTION PLAY: Bull Put Spread**\n"
                       f"**SELL**: ${strike_short} PUT\n"
                       f"**BUY**:  ${strike_long} PUT\n"
                       f"Credit: ~$0.40 | Margin: $100\n"
                       f"Logic: Eat Time Decay (Theta). Profit if Flat or Up.")
                
                notifier.send_alert(f"🦅 Sparse Long Signal: {TICKER}", msg, 0x00ffff)

            # C. EXIT WARNING: Conf >= 0.41 (same as original)
            elif (prev_conf is None or prev_conf < EXIT_WARNING_CONF) and current_conf >= EXIT_WARNING_CONF:
                logging.info(f"⚡ DYNAMIC EXIT SIGNAL TRIGGERED!")
                logging.info(f"   Conf: {current_conf:.2f} (>= {EXIT_WARNING_CONF})")
                
                msg = (f"🦅 **SPARSE REVERSAL: {TICKER}**\n"
                       f"⚡ **DYNAMIC TAKE PROFIT ALERT**\n"
                       f"Price: {curr_price:.2f}\n"
                       f"📈 Conf: **{current_conf:.2f}** (Rising Risk)\n"
                       f"⏰ Time: {t.strftime('%H:%M')}\n\n"
                       f"💡 **Action**: Close Long / Take Profit\n"
                       f"Reason: Model confidence rising on 15m timeframe.")
                
                notifier.send_alert(f"⚡ Sparse Exit Alert: {TICKER}", msg, 0xff00ff)

            else:
                pass
            
            logging.info(f"Scanning... {log_msg}")
            
            # Update State
            prev_conf = current_conf
            
            # Wait (5 minutes between scans)
            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            logging.info("Stopping...")
            break
        except Exception as e:
            logging.error(f"Loop Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()

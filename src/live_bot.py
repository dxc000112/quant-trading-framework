import time
import os
import logging
import traceback
import joblib
import pandas as pd
import numpy as np
import datetime
import yaml
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from src.data_loader import fetch_data
from src.labeling import get_daily_vol
from src.meta_model import get_features
from src.plotting import plot_live_setup
from src.notifier import DiscordNotifier, play_beep

load_dotenv()

# Configure structured logging with file + console output
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("live_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Load config with defaults ──────────────────────────────────────
def _load_bot_config(config_path='config.yaml'):
    defaults = {
        'ticker': 'SPY',
        'interval': '1m',
        'period': '5d',
        'model_path': 'rf_model_short.pkl',
        'confidence_threshold': 0.63,
        'cooldown_seconds': 300,
        'pt_multiplier': 0.4,
        'sl_multiplier': 0.6,
        'breakeven_ratio': 0.5,
        'resample_rule': '5min',
    }
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        bot_cfg = cfg.get('live_bot', {})
        for key in defaults:
            if key in bot_cfg:
                defaults[key] = bot_cfg[key]
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults.")
    return defaults

BOT_CONFIG = _load_bot_config()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
MODEL_PATH = BOT_CONFIG['model_path']
TICKER = BOT_CONFIG['ticker']
INTERVAL = BOT_CONFIG['interval']
PERIOD = BOT_CONFIG['period']

@dataclass
class TradeState:
    entry_price: float
    current_sl: float
    current_tp: float
    volatility: float
    side: int  # 1 for Long, -1 for Short
    start_time: datetime.datetime
    breakeven_triggered: bool = False

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Run src/train.py first.")
        return None
    return joblib.load(MODEL_PATH)

def main():
    logger.info("--- Live Inference Bot Started ---")
    
    # 1. Load Model
    model_data = load_model()
    if not model_data:
        return
        
    clf = model_data['model']
    required_features = model_data['features']
    pt_sl = model_data['pt_sl']
    
    logger.info(f"Model Loaded. Features: {required_features}")
    
    notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
    last_alert_time = 0
    COOLDOWN = BOT_CONFIG['cooldown_seconds']
    CONF_THRESHOLD = BOT_CONFIG['confidence_threshold']
    PT_MULT = BOT_CONFIG['pt_multiplier']
    SL_MULT = BOT_CONFIG['sl_multiplier']
    BE_RATIO = BOT_CONFIG['breakeven_ratio']
    

    trade: Optional[TradeState] = None
    
    while True:
        try:
            logger.info(f"Tick at {datetime.datetime.now().strftime('%H:%M:%S')}")
            
            # Fetch Latest Data
            df = fetch_data(TICKER, period=PERIOD, interval=INTERVAL)
            if df is None:
                time.sleep(60)
                continue
                
            curr_price = df['Close'].iloc[-1]
            
            # --- STATE: MANAGING ACTIVE TRADE ---
            if trade is not None:
                logger.info(f"Monitor Trade | Price: {curr_price:.2f} | SL: {trade.current_sl:.2f} | TP: {trade.current_tp:.2f}")
                
                # Check Exit Conditions (Short Logic: Side -1)
                # Win: Price <= TP
                # Loss: Price >= SL
                
                is_win = False
                is_loss = False
                
                if trade.side == -1: # SHORT
                    if curr_price <= trade.current_tp: is_win = True
                    elif curr_price >= trade.current_sl: is_loss = True
                else: # LONG
                     if curr_price >= trade.current_tp: is_win = True
                     elif curr_price <= trade.current_sl: is_loss = True
                     
                if is_win:
                    logger.info("Trade result: WIN!")
                    notifier.send_alert(f"✅ WIN: {TICKER}", f"Target Hit at {curr_price:.2f}\nEntry: {trade.entry_price:.2f}", 0x00ff00)
                    trade = None
                    last_alert_time = time.time() # Reset cooldown logic if needed, or separate it
                elif is_loss:
                    logger.info("Trade result: LOSS")
                    notifier.send_alert(f"❌ LOSS: {TICKER}", f"Stop Hit at {curr_price:.2f}\nEntry: {trade.entry_price:.2f}", 0xff0000)
                    trade = None
                    last_alert_time = time.time()
                else:
                    # CHECK BREAKEVEN
                    # Condition: 50% to target.
                    # Short: Entry - Price >= 0.5 * (Entry - TP)  OR  Price drops by Vol * 0.3
                    # Distance to TP = 0.6 * Vol. Half is 0.3 * Vol.
                    
                    if not trade.breakeven_triggered:
                        # Calculate distance in favor
                        if trade.side == -1:
                            dist = trade.entry_price - curr_price
                        else:
                            dist = curr_price - trade.entry_price
                            
                        # Threshold: BE_RATIO * PT_MULT * Vol
                        be_threshold = BE_RATIO * PT_MULT * trade.volatility
                        
                        if dist >= be_threshold:
                            logger.info("MOVING TO BREAKEVEN")
                            trade.current_sl = trade.entry_price
                            trade.breakeven_triggered = True
                            
                            notifier.send_alert(f"🔒 Breakeven Locked: {TICKER}", 
                                                f"Price moved {dist:.2f} in favor.\nStop moved to Entry: {trade.entry_price:.2f}", 
                                                0xffff00) # Yellow
                                                
                            # Send Updated Chart
                            # Re-plot with new SL and 'Entry' line
                            history_df = df['Close'].tail(50)
                            barriers = {'pt': trade.current_tp, 'sl': trade.current_sl}
                            extra = {'Entry (BE)': trade.entry_price}
                            
                            # Confidence is whatever created it, or re-infer? Let's just pass N/A or store confidence in trade state.
                            # For now just 0.0 or skip
                            buf = plot_live_setup(curr_price, barriers, history_df, 0.99, extra_lines=extra)
                            notifier.send_chart(buf, "Trade Management: Breakeven Locked")
                            
                    logger.info("Holding position")

            # --- STATE: SCANNING ---
            else:
                logger.debug("Scanning...")
                
                prices = df['Close']
                
                # --- 5-Minute Resampling (Latency Fix) ---
                # We resample the clean 1-min data to 5-min bars
                agg_dict = {
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum',
                    'VWAP': 'mean' 
                }
                df_5m = df.resample(BOT_CONFIG['resample_rule']).agg(agg_dict).dropna()
                
                # 3. Calculate Features on 5m Data
                vol = get_daily_vol(df_5m['Close'])
                X = get_features(df_5m, vol) # Pass 5m df
                
                if not X.empty:
                    latest_X = X.iloc[[-1]]
                    latest_X = latest_X[required_features]
                    
                    prob = clf.predict_proba(latest_X)[0][1]
                    logger.info(f"Confidence: {prob:.2f}")
                    
                    # 5. Logic
                    if prob > CONF_THRESHOLD:
                        current_time = time.time()
                        if current_time - last_alert_time > COOLDOWN:
                            logger.info(f"ENTRY SIGNAL at {curr_price:.2f} (conf={prob:.2f})")
                            
                            curr_vol = vol.iloc[-1]
                            
                            # SHORT LOGIC (Implicit from "Short Sniper" context)
                            # Side = -1
                            side = -1 
                            # If model supports side prediction? No, meta model predicts "Success of Primary Signal".
                            # Here we assume "Short Sniper" model trained on short signals.
                            
                            # Params
                            # Entry: X
                            # SL: X + 0.8 * Vol
                            # TP: X - 0.6 * Vol
                            
                            pt_dist = PT_MULT * curr_vol
                            sl_dist = SL_MULT * curr_vol
                            
                            tp_price = curr_price - pt_dist
                            sl_price = curr_price + sl_dist # Stop is HIGHER for Short
                            
                            # Create Trade
                            trade = TradeState(
                                entry_price=curr_price,
                                current_sl=sl_price,
                                current_tp=tp_price,
                                volatility=curr_vol,
                                side=side,
                                start_time=datetime.datetime.now()
                            )
                            
                            barriers = {'pt': tp_price, 'sl': sl_price}
                            
                            # Live Chart
                            history_df = prices.tail(50)
                            buf = plot_live_setup(curr_price, barriers, history_df, prob)
                            
                            # Alert
                            msg = (f"🔥 SHORT ENTRY ({prob:.0%}) 🔥\n"
                                   f"Price: {curr_price:.2f}\n"
                                   f"Target: {tp_price:.2f}\n"
                                   f"Stop: {sl_price:.2f}")
                            
                            notifier.send_alert(f"Live Signal: {TICKER}", msg, 0x00ff00)
                            notifier.send_chart(buf, "New Short Setup")
                            
                            last_alert_time = current_time
                        else:
                            logger.debug(f"Signal suppressed (cooldown {COOLDOWN}s)")
                    else:
                        logger.debug(f"Below threshold (conf={prob:.2f})")
                else:
                    logger.warning("Not enough data for feature generation.")
            
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}\n{traceback.format_exc()}")
            time.sleep(60)

if __name__ == "__main__":
    main()

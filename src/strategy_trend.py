import pandas as pd
import numpy as np

def check_trend_signal(df: pd.DataFrame) -> int:
    """
    Checks for VWAP + EMA Trend Trend Signal.
    
    Logic:
    1. Calculate VWAP (Intraday from start of DF, assuming DF contains single day or is handled correctly).
       Note: If DF is multi-day, caller should pass only today's data or we reset VWAP.
       For simplicity here, we assume df is passed with adequate history but we might need to strictly calculate VWAP from 9:30.
       However, simplified 'rolling' VWAP or just using the provided buffer is often close enough for short term, 
       BUT for accurate VWAP we need the Anchor at Open. 
       
       We will assume data passed in starts from Market Open or we calculate 'rolling' if that's acceptable.
       Better: valid VWAP requires anchoring. We can try to approximate or expect caller to slice.
       Let's implement a robust VWAP if 'datetime' is available, or simple cumsum if just intraday buffer.
    
    2. EMA 9 & EMA 21.
    
    3. Signal:
       LONG: Price > VWAP & EMA9 > EMA21 & Pullback to EMA9.
       SHORT: Price < VWAP & EMA9 < EMA21 & Pullback to EMA9.
       
    Returns:
        1 (Long)
        -1 (Short)
        0 (Neutral)
    """
    if df is None or df.empty:
        return 0
        
    # Time Filter: Strict Market Hours (09:30 - 16:00 ET)
    # Handling both UTC (Alpaca default) and ET (Local)
    # Ideally, convert to ET for consistency.
    last_ts = df.index[-1]
    
    # Check if tz-aware
    if last_ts.tzinfo is None:
        # Assume UTC if not set, or naive
        pass
    else:
        # Convert to US/Eastern
        try:
             import pytz
             est = pytz.timezone('US/Eastern')
             last_ts = last_ts.astimezone(est)
        except:
             pass

    # Strict Check: 09:30 Start (No Pre-market)
    # User complained about 22:24 CN which is 09:24 ET.
    t_hour = last_ts.hour
    t_min = last_ts.minute
    
    # If before 09:30
    if t_hour < 9 or (t_hour == 9 and t_min < 30):
        # print("🛑 Pre-market Wait")
        return 0
        
    # If after 16:00 (Post-market)
    if t_hour >= 16:
        return 0


    # ... (Indicators Calculation from previous code stays same, assuming df copy happens below) ...
    # Ensure columns are present (Handle Case Sensitivity)
    df = df.copy()
    if 'Close' in df.columns:
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'High': 'high', 'Low': 'low', 'Open': 'open'})
    
    # ... Indicators ...
    if len(df) < 21: return 0
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vp'] = df['tp'] * df['volume']
    if isinstance(df.index, pd.DatetimeIndex):
        df['date'] = df.index.date
        df['cum_vp'] = df.groupby('date')['vp'].cumsum()
        df['cum_vol'] = df.groupby('date')['volume'].cumsum()
        df['vwap'] = df['cum_vp'] / df['cum_vol']
    else:
        df['vwap'] = (df['vp']).cumsum() / df['volume'].cumsum()
    
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # Get Latest
    current_price = df['close'].iloc[-1]
    vwap = df['vwap'].iloc[-1]
    ema_9 = df['ema_9'].iloc[-1]
    ema_21 = df['ema_21'].iloc[-1]

    # Thresholds
    PULLBACK_THRESHOLD = 0.005 

    # --- LONG ONLY (Bull Market Mode) ---
    # Price > VWAP and EMA9 > EMA21
    if current_price > vwap and ema_9 > ema_21:
        # Pullback Check
        dist_pct = (current_price - ema_9) / ema_9
        
        if dist_pct < PULLBACK_THRESHOLD and current_price > ema_21:
            return 1
            
    # --- SHORT DISABLED ---
    # User Request: Disable shorts for Bull Market
    # elif current_price < vwap and ema_9 < ema_21:
    #     dist_pct = (ema_9 - current_price) / current_price
    #     if dist_pct < PULLBACK_THRESHOLD and current_price < ema_21:
    #         return -1

    return 0


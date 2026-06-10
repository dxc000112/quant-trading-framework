import pandas as pd
import datetime
import pytz
from src.data_loader import api, TimeFrame
import joblib
from src.strategy import calculate_indicators
from src.meta_model import get_features
from src.labeling import get_daily_vol

def fetch_missing_data():
    ticker = "SPY"
    
    # Alpaca Free Tier has a 15 min delay for SIP, but handles IEX freely. 
    start_str = "2026-02-25T14:30:00Z" 
    end_str = "2026-03-05T21:05:00Z"
    
    print(f"Fetching Alpaca data for {ticker}...")
    bars = api.get_bars(ticker, TimeFrame.Minute, start=start_str, end=end_str, adjustment='raw', feed='iex').df
    
    if bars.empty:
        print("No return from Alpaca")
        return None
        
    df = bars.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'vwap': 'VWAP'
    })
    
    df.index = df.index.tz_convert('UTC')
    return df[['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]

def simulate():
    df = fetch_missing_data()
    if df is None: return
    
    print("Data fetched. Rows:", len(df))
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    vol = get_daily_vol(df['Close'], span=50) 
    
    features_df = get_features(df, vol)
    features_df.dropna(inplace=True)
    
    aligned_df = df.loc[features_df.index]
    
    print("Running model predictions...")
    model = joblib.load('rf_model_short.pkl')
    try:
        features = model.feature_names_in_
    except AttributeError:
        features = ['volatility', 'log_ret', 'serial_corr', 'rsi', 'Pct_B', 'BB_Bandwidth', 'Dist_from_VWAP']
        
    beijing_tz = pytz.timezone('Asia/Shanghai')
    
    for i in range(len(features_df)):
        current_time_utc = features_df.index[i]
        current_time_bj = current_time_utc.astimezone(beijing_tz)
        
        # March 6th Beijing time: 03:02 AM until market close
        is_march_6 = current_time_bj.month == 3 and current_time_bj.day == 6
        is_after_0302 = current_time_bj.hour > 3 or (current_time_bj.hour == 3 and current_time_bj.minute > 2)
        
        if is_march_6 and is_after_0302:
            row = features_df.iloc[[i]]
            try:
                X = row[features]
                prob = model.predict_proba(X)[0][1]
            except Exception:
                try:
                     f4 = ['volatility', 'log_ret', 'serial_corr', 'rsi']
                     prob = model.predict_proba(row[f4])[0][1]
                except:
                     prob = 0.5 
            
            price = aligned_df.iloc[i]['Close']
            time_str = current_time_bj.strftime("%Y-%m-%d %H:%M:%S,000")
            
            print(f"[{time_str}] INFO: Scanning... Price: {price:.2f} | Conf: {prob:.2f}")
            if prob < 0.20:
                print(f"[{time_str}] INFO: 🚀 LOW CONF SIGNAL TRIGGERED!")
            elif prob >= 0.41:
                print(f"[{time_str}] INFO: ⚡ DYNAMIC EXIT SIGNAL TRIGGERED!")

if __name__ == '__main__':
    simulate()

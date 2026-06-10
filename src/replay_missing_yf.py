import yfinance as yf
import pandas as pd
import datetime
import pytz
import joblib
from src.strategy import calculate_indicators
from src.meta_model import get_features
from src.labeling import get_daily_vol

def simulate_yf():
    ticker = "SPY"
    
    # 1. Fetch Data
    print(f"Fetching yfinance data for {ticker}...")
    df = yf.download(ticker, period="5d", interval="1m", progress=False)
    
    if df.empty:
        print("No return from yfinance")
        return
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    df.index = df.index.tz_convert('UTC')
    
    # 2. Indicators & Features
    df = calculate_indicators(df)
    vol = get_daily_vol(df['Close'], span=50) 
    features_df = get_features(df, vol)
    features_df.dropna(inplace=True)
    
    aligned_df = df.loc[features_df.index]
    
    # 3. Model
    model = joblib.load('rf_model_short.pkl')
    try:
        features = model.feature_names_in_
    except AttributeError:
        features = ['volatility', 'log_ret', 'serial_corr', 'rsi', 'Pct_B', 'BB_Bandwidth', 'Dist_from_VWAP']
        
    beijing_tz = pytz.timezone('Asia/Shanghai')
    
    # 4. Predict & Print
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
            
            price = float(aligned_df.iloc[i]['Close'].iloc[0]) if isinstance(aligned_df.iloc[i]['Close'], pd.Series) else float(aligned_df.iloc[i]['Close'])
            time_str = current_time_bj.strftime("%Y-%m-%d %H:%M:%S,000")
            
            print(f"[{time_str}] INFO: Scanning... Price: {price:.2f} | Conf: {prob:.2f}")
            if prob < 0.20:
                print(f"[{time_str}] INFO: 🚀 LOW CONF SIGNAL TRIGGERED!")
            elif prob >= 0.41:
                print(f"[{time_str}] INFO: ⚡ DYNAMIC EXIT SIGNAL TRIGGERED!")

if __name__ == '__main__':
    simulate_yf()

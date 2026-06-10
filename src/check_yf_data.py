import yfinance as yf
import pandas as pd
import pytz

def check_times():
    ticker = "SPY"
    
    print(f"Fetching yfinance data for {ticker}...")
    df = yf.download(ticker, period="5d", interval="1m", progress=False)
    
    if df.empty:
        print("No return from yfinance")
        return
        
    df.index = df.index.tz_convert('UTC')
    beijing_tz = pytz.timezone('Asia/Shanghai')
    
    print("\nLast 10 rows from yfinance:")
    for idx in df.index[-10:]:
        bj_time = idx.astimezone(beijing_tz)
        print(f"UTC: {idx} | BJ: {bj_time} | Close: {df.loc[idx, 'Close'].values[0]:.2f}")

if __name__ == '__main__':
    check_times()

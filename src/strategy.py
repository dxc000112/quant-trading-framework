import pandas as pd

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators: VWAP and RSI.
    
    Args:
        df (pd.DataFrame): The input dataframe with 'Close', 'High', 'Low', 'Volume'.
        
    Returns:
        pd.DataFrame: DataFrame with added indicator columns.
    """
    if df is None or df.empty:
        return df
        
    # RSI (14)
    rsi_period = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # VWAP (Intraday) — reset per trading session to avoid cross-day dilution
    # VWAP = Cumulative(Price * Volume) / Cumulative(Volume), reset each day
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vol_x_price = typical_price * df['Volume']

    # Group by trading date for proper session-level VWAP
    session = df.index.date
    df['VWAP'] = vol_x_price.groupby(session).cumsum() / df['Volume'].groupby(session).cumsum()
    
    return df

def check_signal(df: pd.DataFrame) -> tuple:
    """
    Checks for Intraday Bullish Momentum: Price > VWAP AND RSI > 50.
    """
    if df is None or df.empty or len(df) < 15: # Need at least 14 candles for RSI
        return False, {}

    # Get the latest row
    latest = df.iloc[-1]
    
    price = latest['Close']
    vwap = latest['VWAP']
    rsi = latest['RSI']

    if pd.isna(vwap) or pd.isna(rsi):
        return False, {}

    # Signal Condition: Price > VWAP AND RSI > 50
    if price > vwap and rsi > 50:
        return True, {
            'price': price,
            'vwap': vwap,
            'rsi': rsi,
            'date': latest.name
        }
    
    return False, {}

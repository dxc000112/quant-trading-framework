def fetch_data(ticker, period="1d", interval="1m", source="auto"):
    """
    Thin compatibility wrapper around the unified market_data package.
    """
    from src.market_data import fetch_history
    return fetch_history(ticker, period=period, interval=interval, source=source)

def fetch_vix_data(period="5d", interval="1m"):
    """
    Fetches VIX proxy data using the unified market_data package.
    """
    from src.market_data import fetch_vix_proxy_data
    return fetch_vix_proxy_data(period=period, interval=interval, source='auto')

if __name__ == "__main__":
    # Test
    print("Testing Unified Market Data Fetch...")
    df = fetch_data("SPY")
    if df is not None:
        print(df.tail())
        print(f"Fetched {len(df)} rows.")

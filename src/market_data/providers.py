import datetime as dt
import os

import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

from src.market_data.cache import MarketDataCache, default_cache_ttl_seconds
from src.market_data.symbols import to_yfinance_symbol

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    ALPACA_API = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
else:
    ALPACA_API = None


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    if 'VWAP' in df.columns:
        return df

    out = df.copy()
    volume = out['Volume'].replace(0, pd.NA).ffill().fillna(1.0)
    typical_price = (out['High'] + out['Low'] + out['Close']) / 3
    cumulative_volume = volume.cumsum().replace(0, pd.NA)
    out['VWAP'] = ((typical_price * volume).cumsum() / cumulative_volume).fillna(out['Close'])
    return out


def normalize_ohlcv_frame(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    out = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'vwap': 'VWAP',
        'adj close': 'Adj Close',
    }).copy()

    required = ['Open', 'High', 'Low', 'Close']
    if any(col not in out.columns for col in required):
        return None

    if 'Volume' not in out.columns:
        out['Volume'] = 1.0

    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep='last')].sort_index()
    out = add_vwap(out)

    return out[['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']]


def resample_ohlcv(df: pd.DataFrame, rule: str):
    if df is None or df.empty:
        return None

    out = df.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'VWAP': 'mean',
    }).dropna()
    return out if not out.empty else None


def period_to_date_range(period: str):
    now = dt.datetime.now(dt.timezone.utc)
    normalized = str(period).lower()

    if normalized.endswith('d'):
        delta = dt.timedelta(days=int(normalized[:-1]))
    elif normalized.endswith('mo'):
        delta = dt.timedelta(days=int(normalized[:-2]) * 30)
    elif normalized.endswith('y'):
        delta = dt.timedelta(days=int(normalized[:-1]) * 365)
    elif normalized == 'max':
        delta = dt.timedelta(days=3650)
    else:
        delta = dt.timedelta(days=30)

    return now - delta, now


def _alpaca_timeframe(interval: str):
    mapping = {
        '1m': TimeFrame.Minute,
        '5m': TimeFrame(5, TimeFrameUnit.Minute),
        '15m': TimeFrame(15, TimeFrameUnit.Minute),
        '1h': TimeFrame.Hour,
        '1d': TimeFrame.Day,
    }
    return mapping.get(interval)


def fetch_alpaca_history(symbol: str, period="1d", interval="1m"):
    if ALPACA_API is None:
        return None

    timeframe = _alpaca_timeframe(interval)
    if timeframe is None:
        return None

    start_dt, end_dt = period_to_date_range(period)
    try:
        bars = ALPACA_API.get_bars(
            symbol,
            timeframe,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            adjustment='raw',
            feed='iex',
        ).df
    except Exception as exc:
        print(f"Failed to fetch data from Alpaca: {exc}")
        return None

    if bars.empty:
        return None

    return normalize_ohlcv_frame(bars)


def fetch_yfinance_history(symbol: str, period="1d", interval="1m"):
    yf_symbol = to_yfinance_symbol(symbol)
    try:
        df = yf.download(
            yf_symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            group_by='column',
            threads=False,
        )
    except Exception as exc:
        print(f"Failed to fetch data from yfinance: {exc}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex) and yf_symbol in df.columns.get_level_values(-1):
        df = df.xs(yf_symbol, axis=1, level=-1)

    return normalize_ohlcv_frame(df)


def provider_order(symbol: str, source: str):
    normalized_source = str(source).lower()

    if normalized_source != 'auto':
        return [normalized_source]

    return ['alpaca', 'yfinance']


def fetch_history(symbol: str, period='1d', interval='1m', source='auto', adjust='qfq', use_cache=True):
    cache = MarketDataCache()

    for provider in provider_order(symbol, source):
        cache_key = {
            'provider': provider,
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'adjust': adjust,
        }

        if use_cache:
            cached = cache.load(cache_key, max_age_seconds=default_cache_ttl_seconds(interval))
            if cached is not None:
                return cached

        if provider == 'alpaca':
            df = fetch_alpaca_history(symbol, period=period, interval=interval)
        elif provider == 'yfinance':
            df = fetch_yfinance_history(symbol, period=period, interval=interval)
        else:
            raise ValueError(f"Unsupported data provider: {provider}")

        if df is not None and not df.empty:
            if use_cache:
                cache.save(cache_key, df)
            return df

    return None


def fetch_vix_proxy_data(period='5d', interval='1m', source='auto'):
    return fetch_history('VIXY', period=period, interval=interval, source=source)

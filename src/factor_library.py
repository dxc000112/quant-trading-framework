import numpy as np
import pandas as pd

DEFAULT_FACTOR_WINDOWS = (
    2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45, 60, 90, 120, 180, 252
)
DEFAULT_FACTOR_LAGS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    aligned_denominator = denominator.replace(0, np.nan)
    return numerator / aligned_denominator


def _normalize_ohlcv(input_df: pd.DataFrame) -> pd.DataFrame:
    if 'Close' not in input_df.columns:
        raise ValueError("Input data must contain a 'Close' column.")

    out = input_df.copy()
    close = out['Close'].astype(float)

    out['Open'] = out.get('Open', close).astype(float)
    out['High'] = out.get('High', close).astype(float)
    out['Low'] = out.get('Low', close).astype(float)
    out['Volume'] = out.get('Volume', pd.Series(1.0, index=out.index)).astype(float)

    return out[['Open', 'High', 'Low', 'Close', 'Volume']]


def generate_factor_library(
    input_df: pd.DataFrame,
    min_factor_count: int = 3000,
    windows=DEFAULT_FACTOR_WINDOWS,
    lags=DEFAULT_FACTOR_LAGS,
) -> pd.DataFrame:
    """
    Generate a dense, multi-family factor library from OHLCV data.

    The output is intentionally large enough to support aggressive factor
    screening for cross-sectional stock selection.
    """
    df = _normalize_ohlcv(input_df)
    close = df['Close']
    open_ = df['Open']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    prev_close = close.shift(1)
    hl2 = (high + low) / 2
    hlc3 = (high + low + close) / 3
    dollar_volume = close * volume

    base_series = {
        'close': close,
        'open': open_,
        'high': high,
        'low': low,
        'hl2': hl2,
        'hlc3': hlc3,
        'volume': volume,
        'dollar_volume': dollar_volume,
        'return_1': close.pct_change(),
        'log_return_1': np.log(close / prev_close),
        'range_pct': _safe_divide(high - low, prev_close.abs()),
        'body_pct': _safe_divide(close - open_, open_.abs()),
        'gap_pct': _safe_divide(open_ - prev_close, prev_close.abs()),
        'upper_shadow_pct': _safe_divide(high - np.maximum(open_, close), prev_close.abs()),
        'lower_shadow_pct': _safe_divide(np.minimum(open_, close) - low, prev_close.abs()),
        'volume_change_1': volume.pct_change(),
        'turnover_change_1': dollar_volume.pct_change(),
    }

    ratio_like = {
        'close', 'open', 'high', 'low', 'hl2', 'hlc3', 'volume', 'dollar_volume'
    }

    feature_map = {}

    for name, series in base_series.items():
        series = series.astype(float)
        series_change = series.diff()

        for lag in lags:
            shifted = series.shift(lag)
            feature_map[f'{name}_lag_delta_{lag}'] = series.diff(lag)
            if name in ratio_like:
                feature_map[f'{name}_lag_return_{lag}'] = _safe_divide(series, shifted) - 1
            else:
                feature_map[f'{name}_lag_gap_{lag}'] = series - shifted

        for window in windows:
            rolling_mean = series.rolling(window).mean()
            rolling_std = series.rolling(window).std()
            rolling_min = series.rolling(window).min()
            rolling_max = series.rolling(window).max()
            rolling_median = series.rolling(window).median()
            rolling_sum = series.rolling(window).sum()
            rolling_abs_mean = series.abs().rolling(window).mean()
            rolling_change_std = series_change.rolling(window).std()
            ema = series.ewm(span=window, adjust=False, min_periods=window).mean()
            range_width = rolling_max - rolling_min

            feature_map[f'{name}_zscore_{window}'] = _safe_divide(series - rolling_mean, rolling_std)
            feature_map[f'{name}_range_pos_{window}'] = _safe_divide(series - rolling_min, range_width)
            feature_map[f'{name}_min_gap_{window}'] = series - rolling_min
            feature_map[f'{name}_max_gap_{window}'] = rolling_max - series
            feature_map[f'{name}_change_vol_{window}'] = rolling_change_std
            feature_map[f'{name}_abs_mean_gap_{window}'] = series.abs() - rolling_abs_mean

            if name in ratio_like:
                feature_map[f'{name}_mean_gap_{window}'] = _safe_divide(series, rolling_mean) - 1
                feature_map[f'{name}_median_gap_{window}'] = _safe_divide(series, rolling_median) - 1
                feature_map[f'{name}_ema_gap_{window}'] = _safe_divide(series, ema) - 1
                feature_map[f'{name}_std_ratio_{window}'] = _safe_divide(rolling_std, rolling_mean.abs())
                feature_map[f'{name}_sum_ratio_{window}'] = _safe_divide(series, rolling_sum.abs())
            else:
                feature_map[f'{name}_mean_gap_{window}'] = series - rolling_mean
                feature_map[f'{name}_median_gap_{window}'] = series - rolling_median
                feature_map[f'{name}_ema_gap_{window}'] = series - ema
                feature_map[f'{name}_std_level_{window}'] = rolling_std
                feature_map[f'{name}_sum_level_{window}'] = rolling_sum

    close_returns = close.pct_change()
    price_sma = {}
    price_ema = {}
    vol_sma = {}
    return_vol = {}

    for window in windows:
        price_sma[window] = close.rolling(window).mean()
        price_ema[window] = close.ewm(span=window, adjust=False, min_periods=window).mean()
        vol_sma[window] = volume.rolling(window).mean()
        return_vol[window] = close_returns.rolling(window).std()

    for short_window in windows:
        for long_window in windows:
            if short_window >= long_window:
                continue
            feature_map[f'close_sma_cross_{short_window}_{long_window}'] = (
                _safe_divide(price_sma[short_window], price_sma[long_window]) - 1
            )
            feature_map[f'close_ema_cross_{short_window}_{long_window}'] = (
                _safe_divide(price_ema[short_window], price_ema[long_window]) - 1
            )
            feature_map[f'volume_regime_{short_window}_{long_window}'] = (
                _safe_divide(vol_sma[short_window], vol_sma[long_window]) - 1
            )
            feature_map[f'return_vol_regime_{short_window}_{long_window}'] = (
                _safe_divide(return_vol[short_window], return_vol[long_window]) - 1
            )

    features = pd.DataFrame(feature_map, index=df.index)
    features = features.replace([np.inf, -np.inf], np.nan)

    if features.shape[1] < min_factor_count:
        raise ValueError(
            f"Generated {features.shape[1]} factors, below the requested minimum of {min_factor_count}."
        )

    return features

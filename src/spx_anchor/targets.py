from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def round_to_increment(values: Iterable[float], increment: float) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if increment <= 0:
        return values
    return np.round(values / increment) * increment


def weighted_quantile(values: Sequence[float], quantiles, sample_weight=None):
    values = np.asarray(values, dtype=float)
    quantiles = np.atleast_1d(np.asarray(quantiles, dtype=float))

    if values.size == 0:
        return np.full(quantiles.shape, np.nan)

    if sample_weight is None:
        sample_weight = np.ones(values.shape[0], dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    mask = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight >= 0)
    values = values[mask]
    sample_weight = sample_weight[mask]
    if values.size == 0:
        return np.full(quantiles.shape, np.nan)

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cumulative = np.cumsum(sample_weight)
    total = cumulative[-1]
    if total <= 0:
        return np.full(quantiles.shape, np.nan)

    normalized = cumulative / total
    result = np.interp(quantiles, normalized, values)
    return result if len(result) > 1 else result[0]


def label_future_distribution(
    future_bars: pd.DataFrame,
    price_bin_size: float = 5.0,
    lower_quantile: float = 0.15,
    upper_quantile: float = 0.85,
):
    if future_bars is None or future_bars.empty:
        raise ValueError("future_bars must contain at least one row.")

    close = pd.to_numeric(future_bars['Close'], errors='coerce').dropna()
    if close.empty:
        raise ValueError("future_bars does not contain usable Close prices.")

    if 'Volume' in future_bars.columns:
        weights = pd.to_numeric(future_bars['Volume'], errors='coerce').reindex(close.index).fillna(1.0)
    else:
        weights = pd.Series(1.0, index=close.index)

    binned = round_to_increment(close.values, price_bin_size)
    grouped = (
        pd.DataFrame({'price': binned, 'weight': weights.values})
        .groupby('price', as_index=True)['weight']
        .sum()
        .sort_values(ascending=False)
    )
    target_price = float(grouped.index[0])

    lower_bound, upper_bound = weighted_quantile(
        close.values,
        [lower_quantile, upper_quantile],
        sample_weight=weights.values,
    )
    lower_bound = float(lower_bound)
    upper_bound = float(upper_bound)
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound

    future_vwap = float(np.average(close.values, weights=weights.values))
    future_close = float(close.iloc[-1])

    return {
        'target_price': target_price,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'future_vwap': future_vwap,
        'future_close': future_close,
    }

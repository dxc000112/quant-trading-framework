from typing import Optional

import numpy as np
import pandas as pd

from src.spx_anchor.config import SpxAnchorSettings
from src.spx_anchor.structure import compute_structure_metrics


RANK_BASE_COLUMNS = [
    'total_gex',
    'front_expiry_gex',
    'gamma_slope',
    'call_concentration',
    'put_concentration',
]


def _time_on_date(ts: pd.Timestamp, clock: str) -> pd.Timestamp:
    hour, minute = (int(part) for part in clock.split(':'))
    return ts.normalize() + pd.Timedelta(hours=hour, minutes=minute)


def _safe_last(series: pd.Series, fallback: float = np.nan) -> float:
    if series is None or series.empty:
        return float(fallback)
    return float(series.iloc[-1])


def percentile_rank(history: pd.Series, value: float) -> float:
    history = pd.to_numeric(history, errors='coerce').dropna()
    if history.empty or not np.isfinite(value):
        return 0.5
    return float((history <= value).mean())


def compute_price_features(
    session_bars: pd.DataFrame,
    as_of,
    settings: Optional[SpxAnchorSettings] = None,
    previous_close: Optional[float] = None,
    futures_bars: Optional[pd.DataFrame] = None,
):
    if session_bars is None or session_bars.empty:
        raise ValueError("session_bars must contain at least one row.")

    settings = settings or SpxAnchorSettings()
    bars = session_bars.sort_index().copy()
    as_of = pd.Timestamp(as_of)

    close = pd.to_numeric(bars['Close'], errors='coerce')
    open_ = pd.to_numeric(bars['Open'], errors='coerce')
    high = pd.to_numeric(bars['High'], errors='coerce')
    low = pd.to_numeric(bars['Low'], errors='coerce')
    volume = pd.to_numeric(bars.get('Volume', pd.Series(1.0, index=bars.index)), errors='coerce').fillna(0.0)

    if 'VWAP' in bars.columns:
        vwap = pd.to_numeric(bars['VWAP'], errors='coerce')
    else:
        typical = (high + low + close) / 3.0
        cumulative_volume = volume.replace(0.0, np.nan).cumsum().replace(0.0, np.nan)
        vwap = ((typical * volume).cumsum() / cumulative_volume).fillna(close)

    session_open_price = float(open_.iloc[0])
    latest_price = float(close.iloc[-1])
    session_high = float(high.max())
    session_low = float(low.min())
    session_range = session_high - session_low
    session_vwap = float(vwap.iloc[-1])

    anchor_start_ts = _time_on_date(as_of, settings.anchor_start_time)
    initial_window = bars[bars.index <= min(as_of, anchor_start_ts)]
    anchor_reference = float(close.loc[initial_window.index].iloc[-1])
    ib_high = float(pd.to_numeric(initial_window['High'], errors='coerce').max())
    ib_low = float(pd.to_numeric(initial_window['Low'], errors='coerce').min())
    ib_mid = (ib_high + ib_low) / 2.0
    ib_width = ib_high - ib_low

    returns = close.pct_change().dropna()
    realized_vol_5m = float(returns.tail(5).std() * np.sqrt(390)) if len(returns) >= 2 else 0.0
    realized_vol_30m = float(returns.tail(30).std() * np.sqrt(390)) if len(returns) >= 2 else 0.0

    volume_5m = float(volume.tail(5).mean()) if not volume.empty else 0.0
    volume_30m = float(volume.tail(30).mean()) if not volume.empty else 0.0
    last_5m_return = float(close.pct_change(5).iloc[-1]) if len(close) >= 6 else 0.0
    last_15m_return = float(close.pct_change(15).iloc[-1]) if len(close) >= 16 else last_5m_return
    ib_breakout = float(latest_price > ib_high or latest_price < ib_low)

    overnight_gap_pct = np.nan
    if previous_close is not None and np.isfinite(previous_close) and previous_close != 0:
        overnight_gap_pct = session_open_price / float(previous_close) - 1.0

    es_return_since_open = np.nan
    es_vwap_distance = np.nan
    spot_es_divergence = np.nan
    if futures_bars is not None and not futures_bars.empty:
        futures = futures_bars.sort_index().copy()
        futures = futures[futures.index <= as_of]
        futures = futures[futures.index.normalize() == as_of.normalize()]
        if not futures.empty and 'Close' in futures.columns:
            es_close = pd.to_numeric(futures['Close'], errors='coerce').dropna()
            if not es_close.empty:
                es_return_since_open = float(es_close.iloc[-1] / es_close.iloc[0] - 1.0)
                if 'VWAP' in futures.columns:
                    es_vwap = pd.to_numeric(futures['VWAP'], errors='coerce').dropna()
                    if not es_vwap.empty:
                        es_vwap_distance = float(es_close.iloc[-1] - es_vwap.iloc[-1])
                spot_es_divergence = float((latest_price / max(session_open_price, 1e-9) - 1.0) - es_return_since_open)

    return {
        'spot_price': latest_price,
        'session_vwap': session_vwap,
        'opening_drive_pct': anchor_reference / session_open_price - 1.0 if session_open_price else 0.0,
        'opening_range_pct': ib_width / max(anchor_reference, 1e-9),
        'drive_efficiency': abs(anchor_reference - session_open_price) / max(ib_width, 1e-9),
        'return_since_anchor_start': latest_price / max(anchor_reference, 1e-9) - 1.0,
        'session_return_pct': latest_price / max(session_open_price, 1e-9) - 1.0,
        'session_range_pct': session_range / max(latest_price, 1e-9),
        'vwap_distance': latest_price - session_vwap,
        'ib_mid_distance': latest_price - ib_mid,
        'ib_width': ib_width,
        'realized_vol_5m': realized_vol_5m,
        'realized_vol_30m': realized_vol_30m,
        'minutes_from_open': float((as_of - _time_on_date(as_of, settings.session_open)).total_seconds() / 60.0),
        'minutes_to_close': float((_time_on_date(as_of, settings.session_close) - as_of).total_seconds() / 60.0),
        'volume_ratio_5m_30m': volume_5m / max(volume_30m, 1.0),
        'overnight_gap_pct': overnight_gap_pct,
        'first_30m_range': ib_width,
        'last_5m_return': last_5m_return,
        'last_15m_return': last_15m_return,
        'ib_breakout': ib_breakout,
        'post_opening_balance_phase': float((as_of - _time_on_date(as_of, settings.session_open)).total_seconds() / 60.0 >= 30.0),
        'es_return_since_open': es_return_since_open,
        'es_vwap_distance': es_vwap_distance,
        'spot_es_divergence': spot_es_divergence,
    }


def compute_pin_trust_features(
    price_features: dict,
    structure,
    event_flags: Optional[dict] = None,
):
    """Estimate whether strike-level pin logic should influence the closing magnet forecast.

    Observable:
    - Spot path, VWAP distance, realized vol, wall locations, and event flags.

    Approximated:
    - Headline shock severity and whether current option positioning is still able to pin spot.

    Inferred:
    - `shock_score`, `shock_state`, and `pin_trust_multiplier`.
    """
    event_flags = event_flags or {}
    headline_flag = bool(event_flags.get('headline_shock_flag', False))
    macro_event_flag = any(bool(event_flags.get(key, False)) for key in ['cpi', 'fomc', 'nfp'])

    spot_price = float(price_features.get('spot_price', getattr(structure, 'spot_price', 0.0)))
    expected_move = max(float(getattr(structure, 'expected_move', 0.0)), 1.0)
    ib_width = max(abs(float(price_features.get('ib_width', 0.0))), 5.0)
    range_scale = max(expected_move, ib_width, 5.0)
    minutes_from_open = max(float(price_features.get('minutes_from_open', 0.0)), 0.0)
    rv_5 = abs(float(price_features.get('realized_vol_5m', 0.0)))
    rv_30 = abs(float(price_features.get('realized_vol_30m', 0.0)))
    vwap_distance = abs(float(price_features.get('vwap_distance', 0.0)))

    magnet_distance = abs(spot_price - float(getattr(structure, 'weighted_gamma_magnet_price', spot_price)))
    pin_distance = abs(spot_price - float(getattr(structure, 'strongest_pin_strike', spot_price)))
    outside_wall_distance = max(
        spot_price - float(getattr(structure, 'call_wall', spot_price)),
        float(getattr(structure, 'put_wall', spot_price)) - spot_price,
        0.0,
    )
    structure_disagreement = abs(
        float(getattr(structure, 'strongest_pin_strike', spot_price)) -
        float(getattr(structure, 'weighted_gamma_magnet_price', spot_price))
    )

    deanchor_score = np.clip(
        max(magnet_distance, pin_distance, outside_wall_distance, 0.75 * vwap_distance) / range_scale - 0.55,
        0.0,
        1.0,
    )
    structure_break_score = np.clip(structure_disagreement / range_scale - 0.25, 0.0, 1.0)

    if rv_30 > 0:
        vol_ratio = rv_5 / max(rv_30, 1e-9)
        vol_spike_score = np.clip((vol_ratio - 1.15) / 0.85, 0.0, 1.0)
        vol_convergence_score = float(np.exp(-abs(vol_ratio - 1.0)))
    else:
        vol_ratio = np.nan
        vol_spike_score = 0.0
        vol_convergence_score = 0.5

    event_score = 1.0 if headline_flag else 0.6 if macro_event_flag else 0.0
    shock_score = np.clip(
        0.35 * event_score +
        0.30 * deanchor_score +
        0.20 * vol_spike_score +
        0.15 * structure_break_score,
        0.0,
        1.0,
    )

    if minutes_from_open < 30.0:
        opening_trust_cap = 0.35
    else:
        opening_trust_cap = 0.35 + 0.65 * np.clip((minutes_from_open - 30.0) / 90.0, 0.0, 1.0)

    recovery_progress = np.clip((minutes_from_open - 30.0) / 120.0, 0.0, 1.0)
    structure_stability = 1.0 - np.clip(vwap_distance / range_scale, 0.0, 1.0)
    recovery_score = (
        recovery_progress *
        vol_convergence_score *
        (1.0 - deanchor_score) *
        (0.55 + 0.45 * structure_stability)
    )

    pin_trust_multiplier = np.clip(
        opening_trust_cap *
        (1.0 - 0.85 * shock_score) *
        (0.55 + 0.45 * recovery_score),
        0.0,
        1.0,
    )
    shock_reprice = float(shock_score >= 0.65 or (headline_flag and deanchor_score >= 0.25))

    if minutes_from_open < 30.0:
        shock_state = 'opening_uncertain'
    elif shock_reprice >= 1.0:
        shock_state = 'shock_active'
    elif pin_trust_multiplier >= 0.60 and vol_convergence_score >= 0.75 and deanchor_score <= 0.25:
        shock_state = 'recovery'
    else:
        shock_state = 'normal'

    return {
        'headline_shock_flag': float(headline_flag),
        'event_shock_context': float(macro_event_flag),
        'event_score': float(event_score),
        'deanchor_score': float(deanchor_score),
        'structure_break_score': float(structure_break_score),
        'vol_ratio_5m_30m': float(vol_ratio) if np.isfinite(vol_ratio) else np.nan,
        'vol_spike_score': float(vol_spike_score),
        'vol_convergence_score': float(vol_convergence_score),
        'opening_trust_cap': float(opening_trust_cap),
        'recovery_score': float(recovery_score),
        'shock_score': float(shock_score),
        'shock_reprice': float(shock_reprice),
        'pin_trust_multiplier': float(pin_trust_multiplier),
        'shock_state': shock_state,
        'shock_state_code': {
            'normal': 0.0,
            'recovery': 1.0,
            'opening_uncertain': 2.0,
            'shock_active': 3.0,
        }[shock_state],
    }


def detect_shock_reprice(
    price_features: dict,
    structure,
    event_flags: Optional[dict] = None,
):
    """Backward-compatible wrapper around the richer pin-trust shock filter."""
    trust_features = compute_pin_trust_features(
        price_features=price_features,
        structure=structure,
        event_flags=event_flags,
    )
    return {
        'headline_shock_flag': trust_features['headline_shock_flag'],
        'event_shock_context': trust_features['event_shock_context'],
        'shock_score': trust_features['shock_score'],
        'shock_reprice': trust_features['shock_reprice'],
    }


def build_feature_snapshot(
    session_bars: pd.DataFrame,
    option_snapshot: pd.DataFrame,
    settings: Optional[SpxAnchorSettings] = None,
    previous_close: Optional[float] = None,
    rank_reference: Optional[pd.DataFrame] = None,
    futures_bars: Optional[pd.DataFrame] = None,
    event_flags: Optional[dict] = None,
):
    settings = settings or SpxAnchorSettings()
    if session_bars is None or session_bars.empty:
        raise ValueError("session_bars must contain at least one row.")

    as_of = pd.Timestamp(session_bars.index[-1])
    price_features = compute_price_features(
        session_bars=session_bars,
        as_of=as_of,
        settings=settings,
        previous_close=previous_close,
        futures_bars=futures_bars,
    )
    structure = compute_structure_metrics(
        option_snapshot=option_snapshot,
        spot_price=price_features['spot_price'],
        as_of=as_of,
        contract_size=settings.contract_size,
        intraday_volume_weight=settings.intraday_volume_weight,
        front_expiry_days=settings.front_expiry_days,
        session_close=settings.session_close,
        intraday_vol_points=(
            abs(float(price_features.get('realized_vol_30m', 0.0))) *
            float(price_features['spot_price']) *
            np.sqrt(max(float(price_features.get('minutes_to_close', 0.0)), 1.0) / 390.0)
        ),
    )
    shock_features = compute_pin_trust_features(price_features, structure, event_flags=event_flags)

    features = {
        'session_date': as_of.normalize(),
        'as_of': as_of,
        **price_features,
        **shock_features,
        'call_wall': structure.call_wall,
        'put_wall': structure.put_wall,
        'long_gamma': structure.long_gamma,
        'gex_pain': structure.gex_pain,
        'gamma_flip': structure.gamma_flip if structure.gamma_flip is not None else np.nan,
        'expected_move': structure.expected_move,
        'total_gex': structure.total_gex,
        'total_gross_gex': structure.total_gross_gex,
        'front_expiry_gex': structure.front_expiry_gex,
        'gamma_slope': structure.gamma_slope,
        'call_concentration': structure.call_concentration,
        'put_concentration': structure.put_concentration,
        'weighted_gamma_center': structure.weighted_gamma_center,
        'weighted_gamma_magnet_price': structure.weighted_gamma_magnet_price,
        'strongest_magnet_strike': structure.strongest_magnet_strike,
        'strongest_pin_strike': structure.strongest_pin_strike,
        'strongest_pin_score': structure.strongest_pin_score,
        'max_pain': structure.max_pain,
        'pin_risk_level': structure.pin_risk_level,
        'pin_risk_score': structure.pin_risk_score,
        'time_to_close_decay': structure.time_to_close_decay,
        'market_regime': structure.market_regime,
        'market_regime_code': {'long_gamma': 1.0, 'short_gamma': -1.0, 'mixed': 0.0}.get(structure.market_regime, 0.0),
        'expiry_gex_0dte': float(structure.expiry_bucket_gex.get('0dte', 0.0)),
        'expiry_gex_1dte': float(structure.expiry_bucket_gex.get('1dte', 0.0)),
        'expiry_gex_weekly': float(structure.expiry_bucket_gex.get('weekly', 0.0)),
        'expiry_gex_standard': float(structure.expiry_bucket_gex.get('standard', 0.0)),
        'wall_width': structure.call_wall - structure.put_wall,
        'dist_call_wall': price_features['spot_price'] - structure.call_wall,
        'dist_put_wall': price_features['spot_price'] - structure.put_wall,
        'dist_long_gamma': price_features['spot_price'] - structure.long_gamma,
        'dist_gex_pain': price_features['spot_price'] - structure.gex_pain,
        'dist_weighted_gamma_center': price_features['spot_price'] - structure.weighted_gamma_center,
        'dist_weighted_gamma_magnet_price': price_features['spot_price'] - structure.weighted_gamma_magnet_price,
        'dist_strongest_magnet_strike': price_features['spot_price'] - structure.strongest_magnet_strike,
        'dist_strongest_pin_strike': price_features['spot_price'] - structure.strongest_pin_strike,
        'dist_max_pain': price_features['spot_price'] - structure.max_pain,
        'dist_pin_risk_level': price_features['spot_price'] - structure.pin_risk_level,
        'spot_inside_walls': float(structure.put_wall <= price_features['spot_price'] <= structure.call_wall),
    }

    for level in structure.gamma_rank_levels[:5]:
        features[f'gamma_rank_{level.rank}_strike'] = level.strike
        features[f'gamma_rank_{level.rank}_score'] = level.score
        features[f'gamma_rank_{level.rank}_magnet_weight'] = level.magnet_weight
        features[f'gamma_rank_{level.rank}_pin_score'] = level.pin_score
        features[f'gamma_rank_{level.rank}_zero_dte_oi'] = level.zero_dte_oi
        features[f'gamma_rank_{level.rank}_gamma_exposure_share'] = level.gamma_exposure_share
        features[f'gamma_rank_{level.rank}_gamma_density'] = level.gamma_density
        features[f'gamma_rank_{level.rank}_oi_concentration'] = level.oi_concentration
        features[f'gamma_rank_{level.rank}_distance_decay'] = level.distance_decay
        features[f'gamma_rank_{level.rank}_time_to_close_decay'] = level.time_to_close_decay
        features[f'gamma_rank_{level.rank}_pin_risk'] = level.pin_risk_score

    for idx in range(len(structure.gamma_rank_levels) + 1, 6):
        features[f'gamma_rank_{idx}_strike'] = np.nan
        features[f'gamma_rank_{idx}_score'] = np.nan
        features[f'gamma_rank_{idx}_magnet_weight'] = np.nan
        features[f'gamma_rank_{idx}_pin_score'] = np.nan
        features[f'gamma_rank_{idx}_zero_dte_oi'] = np.nan
        features[f'gamma_rank_{idx}_gamma_exposure_share'] = np.nan
        features[f'gamma_rank_{idx}_gamma_density'] = np.nan
        features[f'gamma_rank_{idx}_oi_concentration'] = np.nan
        features[f'gamma_rank_{idx}_distance_decay'] = np.nan
        features[f'gamma_rank_{idx}_time_to_close_decay'] = np.nan
        features[f'gamma_rank_{idx}_pin_risk'] = np.nan

    event_flags = event_flags or {}
    for event_name in ['cpi', 'fomc', 'nfp', 'opex', 'month_end', 'quarter_end', 'headline_shock_flag']:
        features[f'event_{event_name}'] = float(bool(event_flags.get(event_name, False)))

    for column in RANK_BASE_COLUMNS:
        rank_column = f'{column}_rank'
        if rank_reference is None or rank_reference.empty or column not in rank_reference.columns:
            rank_value = 0.5
        else:
            rank_value = percentile_rank(rank_reference[column], features[column])
        features[rank_column] = rank_value
        structure.gamma_ranks[rank_column] = rank_value

    return pd.Series(features), structure


def add_historical_rank_features(
    snapshot_panel: pd.DataFrame,
    lookback_rows: int = 800,
    base_columns=None,
) -> pd.DataFrame:
    if snapshot_panel is None or snapshot_panel.empty:
        return snapshot_panel

    base_columns = base_columns or RANK_BASE_COLUMNS
    out = snapshot_panel.sort_values(['as_of']).reset_index(drop=True).copy()

    for column in base_columns:
        if column not in out.columns:
            continue
        rank_column = f'{column}_rank'
        ranks = []
        values = pd.to_numeric(out[column], errors='coerce').to_numpy(dtype=float)
        for idx, value in enumerate(values):
            start = max(0, idx - lookback_rows)
            history = pd.Series(values[start:idx])
            ranks.append(percentile_rank(history, value))
        out[rank_column] = ranks

    return out

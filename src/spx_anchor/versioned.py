from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.spx_anchor.types import AnchorForecast, AnchorStructure


SUPPORTED_VERSIONS = ('v1', 'v2', 'v3')


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _safe_float(value, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def _signed_ratio(row: pd.Series) -> float:
    gross = max(abs(_safe_float(row.get('total_gross_gex', 0.0))), 1.0)
    return _clamp(_safe_float(row.get('total_gex', 0.0)) / gross, -1.0, 1.0)


def _nearest_wall(spot_price: float, put_wall: float, call_wall: float) -> float:
    return float(put_wall if abs(spot_price - put_wall) <= abs(spot_price - call_wall) else call_wall)


def _time_decay(minutes_to_close: float) -> float:
    return _clamp(1.0 - minutes_to_close / 390.0, 0.0, 1.0)


def _range_scale(row: pd.Series) -> float:
    values = [
        abs(_safe_float(row.get('first_30m_range', 0.0))),
        abs(_safe_float(row.get('ib_width', 0.0))),
        abs(_safe_float(row.get('expected_move', 0.0))),
        5.0,
    ]
    return float(max(values))


def _gamma_rank_levels(row: pd.Series, top_n: int = 5) -> List[Tuple[float, float]]:
    levels = []
    for idx in range(1, top_n + 1):
        strike = row.get(f'gamma_rank_{idx}_strike', np.nan)
        score = row.get(f'gamma_rank_{idx}_score', np.nan)
        if pd.notna(strike) and pd.notna(score):
            levels.append((float(strike), max(float(score), 1e-9)))
    return levels


def _weighted_rank_target(row: pd.Series) -> float:
    levels = _gamma_rank_levels(row)
    if not levels:
        return float(row['spot_price'])
    strikes = np.array([level[0] for level in levels], dtype=float)
    weights = np.array([level[1] for level in levels], dtype=float)
    return float(np.average(strikes, weights=weights))


def _concentration_score(row: pd.Series) -> float:
    levels = _gamma_rank_levels(row, top_n=3)
    if not levels:
        return 0.0
    scores = np.array([level[1] for level in levels], dtype=float)
    return float(scores.mean())


def _session_vwap_price(row: pd.Series) -> float:
    """Return the observable session VWAP price from spot and VWAP distance."""
    return float(row['spot_price']) - float(row.get('vwap_distance', 0.0))


def _closing_magnet_components(row: pd.Series) -> Dict[str, float]:
    """Build the structural anchor basket used by the closing magnet model."""
    spot = float(row['spot_price'])
    nearest_wall = _nearest_wall(spot, float(row['put_wall']), float(row['call_wall']))
    return {
        'spot': spot,
        'weighted_gamma_magnet_price': _safe_float(row.get('weighted_gamma_magnet_price', spot), spot),
        'strongest_magnet_strike': _safe_float(row.get('strongest_magnet_strike', spot), spot),
        'weighted_gamma_center': _safe_float(row.get('weighted_gamma_center', spot), spot),
        'rank_target': _weighted_rank_target(row),
        'strongest_pin_strike': _safe_float(row.get('strongest_pin_strike', spot), spot),
        'max_pain': _safe_float(row.get('max_pain', spot), spot),
        'nearest_wall': nearest_wall,
        'session_vwap': _session_vwap_price(row),
    }


def _magnet_core_anchor(components: Dict[str, float]) -> float:
    """Combine the highest-signal structural magnets into one interpretable anchor."""
    return float(np.average(
        [
            components['weighted_gamma_magnet_price'],
            components['strongest_magnet_strike'],
            components['weighted_gamma_center'],
            components['rank_target'],
        ],
        weights=[0.35, 0.25, 0.20, 0.20],
    ))


def _wall_stability_anchor(components: Dict[str, float], v2_target: float) -> float:
    """Blend broad strike structure that often remains relevant even when pinning weakens."""
    return float(np.average(
        [
            v2_target,
            components['nearest_wall'],
            components['max_pain'],
            components['session_vwap'],
        ],
        weights=[0.40, 0.25, 0.20, 0.15],
    ))


def _pin_anchor(components: Dict[str, float]) -> float:
    """Concentrate on the closest strike that is most likely to pin late-day price."""
    return float(np.average(
        [
            components['strongest_pin_strike'],
            components['strongest_magnet_strike'],
            components['max_pain'],
            components['nearest_wall'],
        ],
        weights=[0.45, 0.20, 0.20, 0.15],
    ))


def _version_1_row(row: pd.Series) -> Dict[str, float]:
    spot = float(row['spot_price'])
    call_wall = float(row['call_wall'])
    put_wall = float(row['put_wall'])
    wall_mid = (call_wall + put_wall) / 2.0
    nearest_wall = _nearest_wall(spot, put_wall, call_wall)
    max_pain = float(row['max_pain'])
    expected_move = max(float(row['expected_move']), 5.0)
    wall_width = max(abs(call_wall - put_wall), 5.0)
    signed_ratio = _signed_ratio(row)

    if signed_ratio >= 0:
        target = np.average(
            [wall_mid, max_pain, nearest_wall],
            weights=[0.45, 0.35, 0.20],
        )
        half_width = max(expected_move * 0.55, wall_width * 0.35, 5.0)
        low = max(target - half_width, put_wall - 5.0)
        high = min(target + half_width, call_wall + 5.0)
        if low >= high:
            low, high = target - max(expected_move * 0.45, 5.0), target + max(expected_move * 0.45, 5.0)
    else:
        target = np.average(
            [spot, nearest_wall, max_pain],
            weights=[0.45, 0.35, 0.20],
        )
        half_width = max(expected_move * 0.75, wall_width * 0.40, 7.5)
        low, high = target - half_width, target + half_width

    confidence = _clamp(0.42 + 0.18 * max(signed_ratio, 0.0), 0.18, 0.72)
    return {
        'pred_target_price': float(target),
        'pred_lower_bound': float(low),
        'pred_upper_bound': float(high),
        'confidence_score': float(confidence),
        'version_regime': 'long_gamma' if signed_ratio >= 0 else 'short_gamma',
        'component_primary': float(wall_mid),
        'component_secondary': float(max_pain),
    }


def _version_2_row(row: pd.Series) -> Dict[str, float]:
    v1 = _version_1_row(row)
    spot = float(row['spot_price'])
    weighted_center = float(row['weighted_gamma_center'])
    rank_target = _weighted_rank_target(row)
    max_pain = float(row['max_pain'])
    expected_move = max(float(row['expected_move']), 5.0)
    range_scale = _range_scale(row)
    vwap_distance = float(row.get('vwap_distance', 0.0))
    realized_vol = abs(float(row.get('realized_vol_30m', 0.0)))
    time_decay = _time_decay(float(row.get('minutes_to_close', 390.0)))
    zero_dte_share = _clamp(
        abs(_safe_float(row.get('expiry_gex_0dte', 0.0))) / max(abs(_safe_float(row.get('total_gross_gex', 0.0))), 1.0),
        0.0,
        1.0,
    )
    concentration = _concentration_score(row)

    concentration_target = np.average(
        [rank_target, weighted_center, max_pain],
        weights=[0.45, 0.35, 0.20],
    )
    anchor_weight = _clamp(0.25 + 0.30 * zero_dte_share + 0.20 * time_decay, 0.25, 0.75)
    rv_penalty = _clamp(realized_vol / 0.02, 0.0, 1.25)
    mean_reversion_weight = _clamp(0.18 + 0.15 * (1.0 - rv_penalty), 0.04, 0.28)
    vwap_reversion_target = spot - mean_reversion_weight * vwap_distance

    target = (1.0 - anchor_weight) * float(v1['pred_target_price']) + anchor_weight * concentration_target
    target = 0.85 * target + 0.15 * vwap_reversion_target

    half_width = expected_move * (0.72 - 0.15 * time_decay)
    half_width *= 0.95 - min(concentration / 0.35, 0.20)
    half_width *= 1.0 + 0.55 * rv_penalty
    half_width *= 1.0 - 0.15 * zero_dte_share
    half_width = max(half_width, range_scale * 0.25, 5.0)

    confidence = (
        0.45 +
        0.12 * min(concentration / 0.25, 1.0) +
        0.10 * zero_dte_share +
        0.08 * time_decay -
        0.10 * min(rv_penalty, 1.0)
    )
    confidence = _clamp(confidence, 0.24, 0.84)

    return {
        'pred_target_price': float(target),
        'pred_lower_bound': float(target - half_width),
        'pred_upper_bound': float(target + half_width),
        'confidence_score': float(confidence),
        'version_regime': 'long_gamma' if _signed_ratio(row) >= 0 else 'short_gamma',
        'component_primary': float(concentration_target),
        'component_secondary': float(vwap_reversion_target),
        'zero_dte_share': float(zero_dte_share),
    }


def classify_version_3_regime(row: pd.Series) -> str:
    shock_state = str(row.get('shock_state', 'normal'))
    shock_score = _clamp(_safe_float(row.get('shock_score', 0.0)), 0.0, 1.0)
    pin_trust = _clamp(_safe_float(row.get('pin_trust_multiplier', 0.0)), 0.0, 1.0)

    if shock_state == 'shock_active' or shock_score >= 0.65 or bool(row.get('shock_reprice', 0.0) >= 1.0):
        return 'event_shock'

    signed_ratio = _signed_ratio(row)
    range_scale = _range_scale(row)
    vwap_distance = abs(float(row.get('vwap_distance', 0.0)))
    pin_risk = float(row.get('pin_risk_score', 0.0))
    strongest_pin_score = _clamp(_safe_float(row.get('strongest_pin_score', 0.0)), 0.0, 1.0)
    inside_walls = float(row.get('spot_inside_walls', 0.0)) >= 1.0
    rank_1 = row.get('gamma_rank_1_strike', np.nan)
    near_rank_1 = pd.notna(rank_1) and abs(float(row['spot_price']) - float(rank_1)) <= 0.35 * range_scale
    trend_pressure = (
        abs(float(row.get('last_15m_return', 0.0))) >= 0.0045 or
        vwap_distance >= 0.65 * range_scale or
        abs(float(row.get('spot_es_divergence', 0.0))) >= 0.0015
    )

    if (
        signed_ratio >= 0.10 and
        inside_walls and
        pin_risk >= 0.15 and
        near_rank_1 and
        strongest_pin_score >= 0.20 and
        pin_trust >= 0.45 and
        vwap_distance <= 0.35 * range_scale
    ):
        return 'pin_day'
    if (
        float(row.get('ib_breakout', 0.0)) >= 1.0 and
        trend_pressure and
        float(row.get('drive_efficiency', 0.0)) >= 0.25 and
        pin_trust <= 0.50
    ):
        return 'trend_day'
    if signed_ratio <= -0.10:
        return 'short_gamma'
    return 'long_gamma'


def _build_v3_target(row: pd.Series, regime: str) -> Tuple[float, float, Dict[str, float]]:
    v1 = _version_1_row(row)
    v2 = _version_2_row(row)
    components = _closing_magnet_components(row)
    expected_move = max(float(row['expected_move']), 5.0)
    range_scale = _range_scale(row)
    time_decay = _time_decay(float(row.get('minutes_to_close', 390.0)))
    pin_trust = _clamp(_safe_float(row.get('pin_trust_multiplier', 0.0)), 0.0, 1.0)
    pin_score = _clamp(_safe_float(row.get('strongest_pin_score', 0.0)), 0.0, 1.0)
    shock_score = _clamp(_safe_float(row.get('shock_score', 0.0)), 0.0, 1.0)
    minutes_from_open = _safe_float(row.get('minutes_from_open', 0.0), 0.0)

    magnet_core = _magnet_core_anchor(components)
    wall_anchor = _wall_stability_anchor(components, float(v2['pred_target_price']))
    pin_anchor = _pin_anchor(components)
    structural_center = float(np.average(
        [magnet_core, wall_anchor, float(v1['pred_target_price'])],
        weights=[0.45, 0.40, 0.15],
    ))

    base_pin_weight = {
        'pin_day': 0.60,
        'long_gamma': 0.38,
        'short_gamma': 0.16,
        'trend_day': 0.10,
        'event_shock': 0.05,
    }.get(regime, 0.25)
    wall_weight = {
        'pin_day': 0.35,
        'long_gamma': 0.42,
        'short_gamma': 0.58,
        'trend_day': 0.55,
        'event_shock': 0.60,
    }.get(regime, 0.45)

    effective_pin_weight = _clamp(
        base_pin_weight * pin_trust * (0.40 + 0.60 * pin_score),
        0.0,
        0.75,
    )
    target = (1.0 - wall_weight) * magnet_core + wall_weight * wall_anchor
    target = (1.0 - effective_pin_weight) * target + effective_pin_weight * pin_anchor

    if regime in {'long_gamma', 'pin_day'}:
        vwap_pull = 0.08 * pin_trust
    elif regime == 'event_shock':
        vwap_pull = 0.02
    else:
        vwap_pull = 0.04
    target = (1.0 - vwap_pull) * target + vwap_pull * components['session_vwap']

    width_multiplier = {
        'pin_day': 0.44,
        'long_gamma': 0.56,
        'short_gamma': 0.88,
        'trend_day': 0.82,
        'event_shock': 1.00,
    }.get(regime, 0.70)
    range_floor = {
        'pin_day': 0.20,
        'long_gamma': 0.24,
        'short_gamma': 0.44,
        'trend_day': 0.38,
        'event_shock': 0.52,
    }.get(regime, 0.30)
    half_width = expected_move * width_multiplier
    half_width *= 1.0 - 0.12 * time_decay * pin_trust
    half_width *= 1.0 + 0.45 * shock_score + 0.20 * (1.0 - pin_trust)
    if minutes_from_open < 30.0:
        half_width *= 1.20
    half_width = max(half_width, range_scale * range_floor, 5.0)

    return float(target), float(half_width), {
        'magnet_core': float(magnet_core),
        'wall_anchor': float(wall_anchor),
        'pin_anchor': float(pin_anchor),
        'structural_center': float(structural_center),
        'effective_pin_weight': float(effective_pin_weight),
        'wall_weight': float(wall_weight),
    }


def _version_3_row(row: pd.Series) -> Dict[str, float]:
    regime = classify_version_3_regime(row)
    v1 = _version_1_row(row)
    v2 = _version_2_row(row)
    target, half_width, anchors = _build_v3_target(row, regime)

    expected_move = max(float(row['expected_move']), 5.0)
    concentration = _concentration_score(row)
    time_decay = _time_decay(float(row.get('minutes_to_close', 390.0)))
    pin_trust = _clamp(_safe_float(row.get('pin_trust_multiplier', 0.0)), 0.0, 1.0)
    pin_support = pin_trust * _clamp(_safe_float(row.get('strongest_pin_score', 0.0)) / 0.50, 0.0, 1.0)
    shock_score = _clamp(_safe_float(row.get('shock_score', 0.0)), 0.0, 1.0)
    opening_penalty = 0.16 if _safe_float(row.get('minutes_from_open', 0.0), 0.0) < 30.0 else 0.0
    consensus = 1.0 - min(np.std([v1['pred_target_price'], v2['pred_target_price'], target]) / expected_move, 1.0)
    regime_bonus = {
        'pin_day': 0.14,
        'long_gamma': 0.08,
        'trend_day': -0.04,
        'short_gamma': -0.08,
        'event_shock': -0.18,
    }.get(regime, 0.0)
    confidence = (
        0.36 +
        0.24 * consensus +
        0.18 * min(concentration / 0.25, 1.0) +
        0.10 * time_decay +
        0.14 * pin_support -
        0.20 * shock_score -
        opening_penalty +
        regime_bonus
    )
    confidence = _clamp(confidence, 0.08, 0.95)

    return {
        'pred_target_price': float(target),
        'pred_lower_bound': float(target - half_width),
        'pred_upper_bound': float(target + half_width),
        'confidence_score': float(confidence),
        'version_regime': regime,
        'component_primary': float(anchors['magnet_core']),
        'component_secondary': float(anchors['pin_anchor']),
        'component_structural_center': float(anchors['structural_center']),
        'effective_pin_weight': float(anchors['effective_pin_weight']),
        'wall_weight': float(anchors['wall_weight']),
    }


def predict_version_row(row: pd.Series, version: str = 'v3') -> Dict[str, object]:
    version = str(version).lower()
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported version: {version}")

    if version == 'v1':
        payload = _version_1_row(row)
    elif version == 'v2':
        payload = _version_2_row(row)
    else:
        payload = _version_3_row(row)

    payload['version'] = version
    payload['forecast_low'] = payload['pred_lower_bound']
    payload['forecast_high'] = payload['pred_upper_bound']
    payload['target_price'] = payload['pred_target_price']
    return payload


def predict_version_levels(feature_frame: pd.DataFrame, version: str = 'v3') -> pd.DataFrame:
    if feature_frame is None or feature_frame.empty:
        raise ValueError("feature_frame must contain at least one row.")

    records = [predict_version_row(row, version=version) for _, row in feature_frame.iterrows()]
    return pd.DataFrame(records, index=feature_frame.index)


def _invalidation_conditions(version: str, row: pd.Series, structure: AnchorStructure, regime: str) -> List[str]:
    range_scale = _range_scale(row)
    base = [
        f"Two consecutive 5m closes outside {row['pred_lower_bound']:.1f}-{row['pred_upper_bound']:.1f}",
        f"Call wall / put wall migrate by more than {max(range_scale * 0.35, 10.0):.1f} pts on fresh snapshots",
    ]
    if version in {'v2', 'v3'}:
        base.append("Rank-1 gamma attractor changes twice in a row or weighted gamma center shifts materially")
    if version == 'v3':
        pin_floor = max(0.20, 0.55 * _clamp(_safe_float(row.get('pin_trust_multiplier', 0.0)), 0.0, 1.0))
        regime_map = {
            'pin_day': f"Pin-day assumptions fail if pin trust falls below {pin_floor:.2f} or spot exits the wall corridor persistently",
            'trend_day': "Trend-day magnet invalidates if rank-1 attractor re-forms near spot and VWAP distance compresses back into balance",
            'short_gamma': "Short-gamma magnet invalidates if pin trust recovers sharply and broad wall structure starts reasserting itself",
            'event_shock': "Shock regime invalidates when de-anchor pressure normalizes and pin trust recovers for two consecutive updates",
            'long_gamma': "Long-gamma magnet invalidates if weighted gamma magnet price and nearest wall both migrate away from the current anchor",
        }
        base.append(regime_map.get(regime, "Regime assumptions fail if the state classifier changes on the next update"))
    return base


def _drivers(version: str, row: pd.Series, structure: AnchorStructure, regime: str) -> List[str]:
    drivers = []
    if version == 'v1':
        drivers.append(
            f"V1 anchor uses net GEX sign {_signed_ratio(row):.2f}, wall midpoint {(row['call_wall'] + row['put_wall']) / 2.0:.1f}, and max pain {row['max_pain']:.1f}"
        )
    elif version == 'v2':
        drivers.append(
            f"V2 blends V1 with strike-level concentration, weighted gamma center {structure.weighted_gamma_center:.1f}, and rank-1 attractor {structure.gamma_rank_levels[0].strike:.1f}"
            if structure.gamma_rank_levels else
            f"V2 blends V1 with weighted gamma center {structure.weighted_gamma_center:.1f}"
        )
        drivers.append(
            f"0DTE share { _clamp(abs(_safe_float(row.get('expiry_gex_0dte', 0.0))) / max(abs(_safe_float(row.get('total_gross_gex', 0.0))), 1.0), 0.0, 1.0):.0%}, time decay {_time_decay(float(row.get('minutes_to_close', 390.0))):.0%}, VWAP distance {row.get('vwap_distance', 0.0):.1f}"
        )
    else:
        drivers.append(f"V3 regime classifier selected `{regime}`")
        drivers.append(
            f"Observable inputs: spot {row['spot_price']:.1f}, VWAP {_session_vwap_price(row):.1f}, magnet core {row['component_primary']:.1f}, pin anchor {row['component_secondary']:.1f}"
        )
        drivers.append(
            f"Approximated dealer positioning uses OI + volume size proxy; inferred shock score {row.get('shock_score', 0.0):.2f}, pin trust {row.get('pin_trust_multiplier', 0.0):.2f}, confidence {row['confidence_score']:.0%}"
        )
    return drivers


def build_versioned_forecast(
    feature_row: pd.Series,
    structure: AnchorStructure,
    version: str = 'v3',
) -> AnchorForecast:
    version_row = pd.Series(predict_version_row(feature_row, version=version))
    merged_row = feature_row.copy()
    for key, value in version_row.items():
        merged_row[key] = value

    numeric_snapshot = {}
    for key, value in merged_row.items():
        if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value):
            numeric_snapshot[key] = float(value)

    version = str(version).lower()
    regime = str(version_row.get('version_regime', structure.market_regime))

    baselines = {}
    if version == 'v3':
        v1 = predict_version_row(feature_row, version='v1')
        v2 = predict_version_row(feature_row, version='v2')
        baselines = {
            'baseline_1': {
                'target_price': float(v1['pred_target_price']),
                'weight': 0.33,
                'label': 'Version 1 baseline',
            },
            'baseline_2': {
                'target_price': float(v2['pred_target_price']),
                'weight': 0.33,
                'label': 'Version 2 baseline',
            },
            'baseline_3': {
                'target_price': float(version_row['pred_target_price']),
                'weight': 0.34,
                'label': 'Version 3 regime forecast',
            },
        }

    return AnchorForecast(
        as_of=pd.Timestamp(feature_row['as_of']),
        spot_price=float(feature_row['spot_price']),
        target_price=float(version_row['pred_target_price']),
        lower_bound=float(version_row['pred_lower_bound']),
        upper_bound=float(version_row['pred_upper_bound']),
        confidence=float(version_row['confidence_score']),
        structure=structure,
        feature_snapshot=numeric_snapshot,
        model_name=f'spx-anchor-{version}',
        drivers=_drivers(version, merged_row, structure, regime),
        invalidation_conditions=_invalidation_conditions(version, merged_row, structure, regime),
        baselines=baselines,
        market_regime=regime,
        shock_reprice=bool(regime == 'event_shock' or feature_row.get('shock_reprice', 0.0) >= 1.0),
    )

from typing import Dict

import numpy as np
import pandas as pd

from src.spx_anchor.types import AnchorForecast, AnchorStructure


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _nearest_wall(spot_price: float, put_wall: float, call_wall: float) -> float:
    return float(put_wall if abs(spot_price - put_wall) <= abs(spot_price - call_wall) else call_wall)


def _phase_progress(minutes_from_open: float) -> float:
    return _clamp((minutes_from_open - 30.0) / 120.0, 0.0, 1.0)


def _baseline_1(row: pd.Series) -> Dict[str, float]:
    spot = float(row['spot_price'])
    gex_pain = float(row['gex_pain'])
    long_gamma = float(row['long_gamma'])
    call_wall = float(row['call_wall'])
    put_wall = float(row['put_wall'])
    wall_mid = (call_wall + put_wall) / 2.0
    nearest_wall = _nearest_wall(spot, put_wall, call_wall)
    regime = row.get('market_regime', 'mixed')
    inside_walls = float(row.get('spot_inside_walls', 0.0))

    if regime == 'long_gamma':
        target = np.average(
            [gex_pain, long_gamma, wall_mid, nearest_wall],
            weights=[0.30, 0.30, 0.20, 0.20 if inside_walls else 0.10],
        )
    elif regime == 'short_gamma':
        target = np.average(
            [spot, gex_pain, nearest_wall],
            weights=[0.40, 0.35, 0.25],
        )
    else:
        target = np.average(
            [spot, gex_pain, wall_mid],
            weights=[0.25, 0.45, 0.30],
        )

    return {
        'target': float(target),
        'explanation_score': float(1.0 - min(abs(spot - gex_pain) / max(float(row['expected_move']), 1.0), 1.0)),
    }


def _baseline_2(row: pd.Series) -> Dict[str, float]:
    max_pain = float(row['max_pain'])
    pin_risk_level = float(row['pin_risk_level'])
    ib_mid = float(row['spot_price'] - row.get('ib_mid_distance', 0.0))
    phase_progress = _phase_progress(float(row.get('minutes_from_open', 30.0)))
    shock_reprice = bool(row.get('shock_reprice', 0.0) >= 1.0)

    target = np.average(
        [max_pain, pin_risk_level, ib_mid],
        weights=[0.50 + 0.10 * phase_progress, 0.35, 0.15 - 0.05 * phase_progress],
    )
    if shock_reprice:
        target = np.average([target, float(row['spot_price'])], weights=[0.65, 0.35])

    return {
        'target': float(target),
        'explanation_score': float(0.5 + 0.5 * min(float(row.get('pin_risk_score', 0.0)), 1.0)),
    }


def _baseline_3(row: pd.Series) -> Dict[str, float]:
    spot = float(row['spot_price'])
    weighted_center = float(row['weighted_gamma_center'])
    regime = row.get('market_regime', 'mixed')
    shock_reprice = bool(row.get('shock_reprice', 0.0) >= 1.0)
    minutes_to_close = max(float(row.get('minutes_to_close', 0.0)), 1.0)
    time_factor = _clamp(1.0 - minutes_to_close / 390.0, 0.0, 1.0)

    if regime == 'long_gamma':
        reversion_strength = 0.55 + 0.20 * time_factor
    elif regime == 'short_gamma':
        reversion_strength = 0.15 + 0.10 * time_factor
    else:
        reversion_strength = 0.35 + 0.10 * time_factor

    if shock_reprice:
        reversion_strength *= 0.55

    target = spot + reversion_strength * (weighted_center - spot)
    return {
        'target': float(target),
        'explanation_score': float(reversion_strength),
    }


def _baseline_weights(row: pd.Series) -> Dict[str, float]:
    phase_progress = _phase_progress(float(row.get('minutes_from_open', 30.0)))
    shock_reprice = bool(row.get('shock_reprice', 0.0) >= 1.0)
    regime = row.get('market_regime', 'mixed')

    weights = {
        'baseline_1': 0.38 + 0.05 * (1.0 - phase_progress),
        'baseline_2': 0.32 + 0.05 * phase_progress,
        'baseline_3': 0.30,
    }

    if regime == 'short_gamma':
        weights['baseline_3'] -= 0.10
        weights['baseline_1'] += 0.05
        weights['baseline_2'] += 0.05
    elif regime == 'long_gamma':
        weights['baseline_3'] += 0.05
        weights['baseline_1'] -= 0.02
        weights['baseline_2'] -= 0.03

    if shock_reprice:
        weights['baseline_3'] -= 0.08
        weights['baseline_2'] -= 0.04
        weights['baseline_1'] += 0.12

    total = sum(max(weight, 0.01) for weight in weights.values())
    return {key: max(value, 0.01) / total for key, value in weights.items()}


def predict_baseline_row(row: pd.Series) -> Dict[str, object]:
    baseline_1 = _baseline_1(row)
    baseline_2 = _baseline_2(row)
    baseline_3 = _baseline_3(row)
    weights = _baseline_weights(row)

    baseline_targets = {
        'baseline_1': baseline_1['target'],
        'baseline_2': baseline_2['target'],
        'baseline_3': baseline_3['target'],
    }
    spot = float(row['spot_price'])
    shock_reprice = bool(row.get('shock_reprice', 0.0) >= 1.0)
    regime = row.get('market_regime', 'mixed')
    expected_move = max(float(row.get('expected_move', 0.0)), 1.0)
    if pd.notna(row.get('gamma_rank_1_strike', np.nan)) and pd.notna(row.get('gamma_rank_2_strike', np.nan)):
        strike_step = abs(float(row['gamma_rank_1_strike']) - float(row['gamma_rank_2_strike']))
    else:
        strike_step = 5.0
    strike_step = max(strike_step, 5.0)

    target = sum(baseline_targets[name] * weights[name] for name in baseline_targets)
    if shock_reprice:
        target = np.average([target, spot], weights=[0.75, 0.25])

    dispersion = float(np.std(list(baseline_targets.values())))
    top_scores = [float(row.get(f'gamma_rank_{idx}_score', np.nan)) for idx in range(1, 4)]
    top_scores = [score for score in top_scores if np.isfinite(score)]
    concentration = float(np.mean(top_scores)) if top_scores else 0.0

    phase_progress = _phase_progress(float(row.get('minutes_from_open', 30.0)))
    width = expected_move * (0.85 - 0.15 * phase_progress)
    width *= 0.90 if regime == 'long_gamma' else 1.15 if regime == 'mixed' else 1.40
    width *= 1.35 if shock_reprice else 1.0
    width *= 1.0 + min(dispersion / max(expected_move, 1.0), 1.0) * 0.6
    width = max(width, strike_step)

    low = float(target - width / 2.0)
    high = float(target + width / 2.0)
    if regime == 'long_gamma' and not shock_reprice:
        low = max(low, float(row['put_wall']) - strike_step)
        high = min(high, float(row['call_wall']) + strike_step)
        if low >= high:
            low = target - strike_step / 2.0
            high = target + strike_step / 2.0

    consensus_score = 1.0 - min(dispersion / max(expected_move, 1.0), 1.0)
    time_score = 0.45 + 0.25 * phase_progress
    regime_bonus = 0.08 if regime == 'long_gamma' else -0.08 if regime == 'short_gamma' else 0.0
    confidence = (
        0.45 * consensus_score +
        0.25 * min(concentration / 0.25, 1.0) +
        0.20 * time_score +
        0.10 * np.mean([baseline_1['explanation_score'], baseline_2['explanation_score'], baseline_3['explanation_score']]) +
        regime_bonus -
        (0.20 if shock_reprice else 0.0)
    )
    confidence = _clamp(float(confidence), 0.05, 0.95)

    return {
        'pred_target_price': float(target),
        'pred_lower_bound': float(low),
        'pred_upper_bound': float(high),
        'confidence_score': confidence,
        'shock_reprice': float(shock_reprice),
        'baseline_1_target': float(baseline_1['target']),
        'baseline_2_target': float(baseline_2['target']),
        'baseline_3_target': float(baseline_3['target']),
        'baseline_1_weight': float(weights['baseline_1']),
        'baseline_2_weight': float(weights['baseline_2']),
        'baseline_3_weight': float(weights['baseline_3']),
    }


def predict_baseline_levels(feature_frame: pd.DataFrame) -> pd.DataFrame:
    if feature_frame is None or feature_frame.empty:
        raise ValueError("feature_frame must contain at least one row.")

    records = [predict_baseline_row(row) for _, row in feature_frame.iterrows()]
    return pd.DataFrame(records, index=feature_frame.index)


def _build_invalidation_conditions(row: pd.Series, structure: AnchorStructure):
    band_break = max(float(structure.expected_move) * 0.35, 10.0)
    return [
        f"Two consecutive 5m closes outside {_clamp(row['pred_lower_bound'], -1e9, 1e9):.1f}-{_clamp(row['pred_upper_bound'], -1e9, 1e9):.1f}",
        f"Spot breaks call wall or put wall by more than {band_break:.1f} pts",
        "Headline shock flag triggers or realized vol spikes into a fresh reprice regime",
        "Market regime flips between long_gamma and short_gamma or gamma flip moves through spot",
    ]


def _build_drivers(row: pd.Series, structure: AnchorStructure):
    levels = []
    for level in structure.gamma_rank_levels[:3]:
        levels.append(f"R{level.rank} {level.strike:.1f} (score {level.score:.3f})")

    drivers = [
        f"Baseline 1 {row['baseline_1_target']:.1f}, Baseline 2 {row['baseline_2_target']:.1f}, Baseline 3 {row['baseline_3_target']:.1f}",
        f"Weights {row['baseline_1_weight']:.0%}/{row['baseline_2_weight']:.0%}/{row['baseline_3_weight']:.0%} across GEX+walls, max-pain/pin, weighted gamma center",
        f"Top attractor strikes: {', '.join(levels) if levels else 'n/a'}",
        f"Regime {structure.market_regime} with weighted gamma center {structure.weighted_gamma_center:.1f} and max pain {structure.max_pain:.1f}",
    ]
    if row.get('shock_reprice', 0.0) >= 1.0:
        drivers.append("Shock reprice mode is active, so the forecast leans closer to live spot and carries a wider range")
    return drivers


def build_baseline_forecast(feature_row: pd.Series, structure: AnchorStructure) -> AnchorForecast:
    baseline_row = pd.Series(predict_baseline_row(feature_row))
    merged_row = feature_row.copy()
    for key, value in baseline_row.items():
        merged_row[key] = value

    numeric_snapshot = {}
    for key, value in merged_row.items():
        if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(value):
            numeric_snapshot[key] = float(value)

    baselines = {
        'baseline_1': {
            'target_price': float(merged_row['baseline_1_target']),
            'weight': float(merged_row['baseline_1_weight']),
            'label': 'Net GEX + call/put wall',
        },
        'baseline_2': {
            'target_price': float(merged_row['baseline_2_target']),
            'weight': float(merged_row['baseline_2_weight']),
            'label': 'Max pain / pin risk',
        },
        'baseline_3': {
            'target_price': float(merged_row['baseline_3_target']),
            'weight': float(merged_row['baseline_3_weight']),
            'label': 'Mean reversion to weighted gamma center',
        },
    }

    return AnchorForecast(
        as_of=pd.Timestamp(feature_row['as_of']),
        spot_price=float(feature_row['spot_price']),
        target_price=float(merged_row['pred_target_price']),
        lower_bound=float(merged_row['pred_lower_bound']),
        upper_bound=float(merged_row['pred_upper_bound']),
        confidence=float(merged_row['confidence_score']),
        structure=structure,
        feature_snapshot=numeric_snapshot,
        model_name='baseline-strike-concentration-engine',
        drivers=_build_drivers(merged_row, structure),
        invalidation_conditions=_build_invalidation_conditions(merged_row, structure),
        baselines=baselines,
        market_regime=structure.market_regime,
        shock_reprice=bool(merged_row['shock_reprice'] >= 1.0),
    )

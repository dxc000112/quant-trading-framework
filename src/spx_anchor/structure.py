from typing import Optional

import numpy as np
import pandas as pd

from src.spx_anchor.types import AnchorStructure, GammaRankLevel


OPTION_RENAME_MAP = {
    'type': 'option_type',
    'right': 'option_type',
    'oi': 'open_interest',
}


def _time_on_date(ts: pd.Timestamp, clock: str) -> pd.Timestamp:
    hour, minute = (int(part) for part in clock.split(':'))
    return ts.normalize() + pd.Timedelta(hours=hour, minutes=minute)


def _time_to_close_decay(as_of: pd.Timestamp, session_close: str, full_session_minutes: float = 390.0) -> float:
    close_ts = _time_on_date(as_of, session_close)
    minutes_to_close = max((close_ts - as_of).total_seconds() / 60.0, 1.0)
    return float(np.clip(1.0 - minutes_to_close / full_session_minutes, 0.0, 1.0))


def _is_standard_monthly(expiry: pd.Timestamp) -> bool:
    expiry = pd.Timestamp(expiry)
    return expiry.weekday() == 4 and 15 <= expiry.day <= 21


def _expiry_bucket(expiry: pd.Timestamp, as_of: pd.Timestamp) -> str:
    expiry = pd.Timestamp(expiry)
    as_of = pd.Timestamp(as_of)
    dte = max((expiry.normalize() - as_of.normalize()).days, 0)
    if dte == 0:
        return '0dte'
    if dte == 1:
        return '1dte'
    if _is_standard_monthly(expiry):
        return 'standard'
    return 'weekly'


def normalize_option_snapshot(options: pd.DataFrame) -> pd.DataFrame:
    if options is None or options.empty:
        raise ValueError("options must contain at least one row.")

    out = options.rename(columns=OPTION_RENAME_MAP).copy()
    required = {'as_of', 'expiry', 'option_type', 'strike', 'gamma'}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"options is missing required columns: {sorted(missing)}")

    out['as_of'] = pd.to_datetime(out['as_of'])
    out['expiry'] = pd.to_datetime(out['expiry'])
    out['option_type'] = out['option_type'].astype(str).str.upper().str[0]
    out = out[out['option_type'].isin(['C', 'P'])].copy()

    for column in ['strike', 'gamma', 'open_interest', 'volume', 'iv', 'delta', 'vanna', 'charm']:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors='coerce')

    out = out.dropna(subset=['as_of', 'expiry', 'strike', 'gamma']).copy()
    out['open_interest'] = out['open_interest'].fillna(0.0).clip(lower=0.0)
    out['volume'] = out['volume'].fillna(0.0).clip(lower=0.0)
    days_to_expiry = (out['expiry'].dt.normalize() - out['as_of'].dt.normalize()).dt.days.clip(lower=0)
    standard_monthly = (out['expiry'].dt.weekday == 4) & out['expiry'].dt.day.between(15, 21)
    out['expiry_bucket'] = np.where(
        days_to_expiry.eq(0),
        '0dte',
        np.where(days_to_expiry.eq(1), '1dte', np.where(standard_monthly, 'standard', 'weekly')),
    )
    return out.sort_values(['as_of', 'expiry', 'strike', 'option_type']).reset_index(drop=True)


def ensure_normalized_option_snapshot(options: pd.DataFrame) -> pd.DataFrame:
    if options is None or options.empty:
        raise ValueError("options must contain at least one row.")

    required = {'as_of', 'expiry', 'option_type', 'strike', 'gamma', 'expiry_bucket'}
    if required.issubset(options.columns):
        return options.copy()
    return normalize_option_snapshot(options)


def latest_snapshot_as_of(
    options: pd.DataFrame,
    as_of,
    max_staleness_minutes: int = 10,
) -> Optional[pd.DataFrame]:
    if options is None or options.empty:
        return None

    out = ensure_normalized_option_snapshot(options)
    as_of = pd.Timestamp(as_of)
    same_day = out[out['as_of'].dt.normalize() == as_of.normalize()]
    same_day = same_day[same_day['as_of'] <= as_of]
    if same_day.empty:
        return None

    latest_ts = same_day['as_of'].max()
    staleness = as_of - latest_ts
    if staleness > pd.Timedelta(minutes=max_staleness_minutes):
        return None

    return same_day[same_day['as_of'] == latest_ts].copy()


def _expected_move(snapshot: pd.DataFrame, spot_price: float, as_of: pd.Timestamp, session_close: str) -> float:
    if snapshot is None or snapshot.empty:
        return 0.0

    close_ts = _time_on_date(as_of, session_close)
    minutes_to_close = max((close_ts - as_of).total_seconds() / 60.0, 1.0)

    atm = snapshot.iloc[(snapshot['strike'] - spot_price).abs().argsort()[:4]]
    atm_iv = pd.to_numeric(atm.get('iv'), errors='coerce').replace(0.0, np.nan).dropna().mean()

    strike_grid = np.diff(np.sort(snapshot['strike'].dropna().unique()))
    strike_step = float(np.median(strike_grid)) if len(strike_grid) else 5.0

    if pd.notna(atm_iv) and atm_iv > 0:
        move = spot_price * float(atm_iv) * np.sqrt(minutes_to_close / (390.0 * 252.0))
        return float(max(move, strike_step))

    return float(strike_step)


def _gamma_flip(net_by_strike: pd.Series, spot_price: float):
    if net_by_strike.empty or len(net_by_strike) < 2:
        return None

    signs = np.sign(net_by_strike.to_numpy())
    strikes = net_by_strike.index.to_numpy(dtype=float)
    changes = np.where(signs[:-1] * signs[1:] < 0)[0]
    if changes.size == 0:
        return None

    candidate_levels = []
    for idx in changes:
        left_strike = strikes[idx]
        right_strike = strikes[idx + 1]
        left_val = net_by_strike.iloc[idx]
        right_val = net_by_strike.iloc[idx + 1]
        if right_val == left_val:
            candidate_levels.append((left_strike + right_strike) / 2.0)
            continue
        weight = abs(left_val) / (abs(left_val) + abs(right_val))
        candidate_levels.append(left_strike + weight * (right_strike - left_strike))

    return float(min(candidate_levels, key=lambda level: abs(level - spot_price)))


def _max_pain_strike(by_strike: pd.DataFrame) -> float:
    if by_strike.empty:
        raise ValueError("by_strike must contain at least one strike.")

    strikes = by_strike.index.to_numpy(dtype=float)
    call_oi = by_strike['call_oi'].to_numpy(dtype=float)
    put_oi = by_strike['put_oi'].to_numpy(dtype=float)

    payout = []
    for candidate in strikes:
        call_payout = np.maximum(candidate - strikes, 0.0) * call_oi
        put_payout = np.maximum(strikes - candidate, 0.0) * put_oi
        payout.append(call_payout.sum() + put_payout.sum())

    return float(strikes[int(np.argmin(payout))])


def _market_regime(total_gex: float, total_gross_gex: float) -> str:
    if total_gross_gex <= 0:
        return 'mixed'

    ratio = total_gex / total_gross_gex
    if ratio >= 0.15:
        return 'long_gamma'
    if ratio <= -0.15:
        return 'short_gamma'
    return 'mixed'


def _compute_weighted_gamma_magnet(
    by_strike: pd.DataFrame,
    spot_price: float,
    expected_move: float,
    strike_step: float,
    as_of: pd.Timestamp,
    session_close: str,
):
    if by_strike.empty:
        raise ValueError("by_strike must contain at least one strike.")

    total_gross_gex = max(float(by_strike['gross_gex'].sum()), 1.0)
    total_open_interest = max(float(by_strike['open_interest'].sum()), 1.0)
    time_decay = _time_to_close_decay(as_of, session_close)

    # As the close approaches, nearby strikes should dominate more strongly.
    distance_scale = max(
        strike_step,
        expected_move * (0.45 + 0.85 * np.sqrt(max(1.0 - time_decay, 1e-9))),
    )

    gamma_weight = 0.52 - 0.10 * time_decay
    oi_weight = 0.20 + 0.10 * time_decay
    distance_weight = 0.28

    out = by_strike.copy()
    out['gamma_exposure_share'] = out['gross_gex'] / total_gross_gex
    out['oi_concentration'] = out['open_interest'] / total_open_interest
    out['distance_decay'] = np.exp(
        -np.abs(out.index.to_numpy(dtype=float) - spot_price) / max(distance_scale, strike_step)
    )
    out['time_to_close_decay'] = time_decay
    out['magnet_weight'] = (
        gamma_weight * out['gamma_exposure_share'] +
        oi_weight * out['oi_concentration'] +
        distance_weight * out['distance_decay']
    )
    total_magnet_weight = max(float(out['magnet_weight'].sum()), 1e-9)
    out['magnet_weight'] = out['magnet_weight'] / total_magnet_weight

    weighted_gamma_magnet_price = float(np.average(
        out.index.to_numpy(dtype=float),
        weights=np.maximum(out['magnet_weight'].to_numpy(dtype=float), 1e-9),
    ))
    strongest_magnet_strike = float(out['magnet_weight'].idxmax())

    top_levels = out.sort_values('magnet_weight', ascending=False).head(5)
    gamma_rank_levels = [
        GammaRankLevel(
            rank=rank,
            strike=float(strike),
            score=float(row['magnet_weight']),
            magnet_weight=float(row['magnet_weight']),
            pin_score=0.0,
            gross_gex=float(row['gross_gex']),
            net_gex=float(row['net_gex']),
            call_gex=float(row['call_gex']),
            put_gex=float(row['put_gex']),
            open_interest=float(row['open_interest']),
            size_proxy=float(row['size_proxy']),
            zero_dte_oi=0.0,
            gamma_exposure_share=float(row['gamma_exposure_share']),
            gamma_density=0.0,
            oi_concentration=float(row['oi_concentration']),
            distance_decay=float(row['distance_decay']),
            time_to_close_decay=float(row['time_to_close_decay']),
            pin_risk_score=float(row['pin_risk_score']),
        )
        for rank, (strike, row) in enumerate(top_levels.iterrows(), start=1)
    ]

    return out, weighted_gamma_magnet_price, strongest_magnet_strike, time_decay, gamma_rank_levels


def _compute_pin_surface(
    by_strike: pd.DataFrame,
    zero_dte_by_strike: pd.DataFrame,
    spot_price: float,
    expected_move: float,
    strike_step: float,
    as_of: pd.Timestamp,
    session_close: str,
    intraday_vol_points: Optional[float] = None,
):
    if by_strike.empty:
        raise ValueError("by_strike must contain at least one strike.")

    out = by_strike.copy()
    zero_dte = zero_dte_by_strike.reindex(out.index).fillna(0.0)
    out['zero_dte_open_interest'] = zero_dte.get('open_interest', pd.Series(0.0, index=out.index))
    out['zero_dte_call_oi'] = zero_dte.get('call_oi', pd.Series(0.0, index=out.index))
    out['zero_dte_put_oi'] = zero_dte.get('put_oi', pd.Series(0.0, index=out.index))

    paired_0dte_oi = np.sqrt(out['zero_dte_call_oi'] * out['zero_dte_put_oi'])
    if float(paired_0dte_oi.max()) <= 0:
        paired_0dte_oi = out['zero_dte_open_interest']
    out['zero_dte_oi_proxy'] = paired_0dte_oi
    out['oi_concentration_pin'] = paired_0dte_oi / max(float(paired_0dte_oi.max()), 1.0)

    strikes = out.index.to_numpy(dtype=float)
    gamma_base = out['gross_gex'].to_numpy(dtype=float)
    bandwidth = max(strike_step * 1.5, expected_move * 0.30)
    kernel = np.exp(-0.5 * ((strikes[:, None] - strikes[None, :]) / max(bandwidth, strike_step)) ** 2)
    gamma_density_raw = kernel @ gamma_base
    out['gamma_density'] = gamma_density_raw / max(float(np.max(gamma_density_raw)), 1.0)

    time_decay = _time_to_close_decay(as_of, session_close)
    distance_scale = max(
        strike_step,
        max(expected_move, strike_step) * (0.90 - 0.45 * time_decay),
    )
    out['distance_decay_pin'] = np.exp(-((np.abs(strikes - spot_price) / max(distance_scale, strike_step)) ** 2))

    if intraday_vol_points is None:
        intraday_vol_points = expected_move
    intraday_vol_points = max(float(intraday_vol_points), strike_step)
    vol_term = 1.0 / (1.0 + intraday_vol_points / max(expected_move, strike_step))
    time_term = 0.30 + 0.70 * np.sqrt(max(time_decay, 0.0))

    out['pin_score'] = (
        0.40 * out['distance_decay_pin'] +
        0.25 * out['oi_concentration_pin'] +
        0.25 * out['gamma_density'] +
        0.10 * time_term
    ) * vol_term
    out['pin_score'] = out['pin_score'].clip(lower=0.0, upper=1.0)

    strongest_pin_strike = float(out['pin_score'].idxmax())
    strongest_pin_score = float(out['pin_score'].max())
    return out, strongest_pin_strike, strongest_pin_score


def compute_structure_metrics(
    option_snapshot: pd.DataFrame,
    spot_price: float,
    as_of,
    contract_size: int = 100,
    intraday_volume_weight: float = 0.25,
    front_expiry_days: int = 1,
    session_close: str = '16:00',
    intraday_vol_points: Optional[float] = None,
) -> AnchorStructure:
    snapshot = ensure_normalized_option_snapshot(option_snapshot)
    as_of = pd.Timestamp(as_of)

    days_to_expiry = (snapshot['expiry'].dt.normalize() - as_of.normalize()).dt.days.clip(lower=0)
    included = snapshot[days_to_expiry <= front_expiry_days].copy()
    if included.empty:
        included = snapshot.copy()

    included['size_proxy'] = included['open_interest'] + intraday_volume_weight * included['volume']
    included['gross_gex'] = included['gamma'].abs() * included['size_proxy'] * contract_size * (spot_price ** 2) * 0.01
    included['signed_gex'] = np.where(
        included['option_type'].eq('C'),
        included['gross_gex'],
        -included['gross_gex'],
    )
    expiry_bucket_gex = included.groupby('expiry_bucket')['gross_gex'].sum().to_dict()
    included['call_gex'] = np.where(included['option_type'].eq('C'), included['gross_gex'], 0.0)
    included['put_gex'] = np.where(included['option_type'].eq('P'), included['gross_gex'], 0.0)
    included['call_oi'] = np.where(included['option_type'].eq('C'), included['open_interest'], 0.0)
    included['put_oi'] = np.where(included['option_type'].eq('P'), included['open_interest'], 0.0)
    included['call_size_proxy'] = np.where(included['option_type'].eq('C'), included['size_proxy'], 0.0)
    included['put_size_proxy'] = np.where(included['option_type'].eq('P'), included['size_proxy'], 0.0)

    by_strike = included.groupby('strike').agg(
        call_gex=('call_gex', 'sum'),
        put_gex=('put_gex', 'sum'),
        net_gex=('signed_gex', 'sum'),
        gross_gex=('gross_gex', 'sum'),
        call_oi=('call_oi', 'sum'),
        put_oi=('put_oi', 'sum'),
        open_interest=('open_interest', 'sum'),
        size_proxy=('size_proxy', 'sum'),
        call_size_proxy=('call_size_proxy', 'sum'),
        put_size_proxy=('put_size_proxy', 'sum'),
    ).sort_index()
    if by_strike.empty:
        raise ValueError("option_snapshot did not produce any usable strike-level structure.")

    zero_dte_snapshot = included[included['expiry_bucket'].eq('0dte')].copy()
    if zero_dte_snapshot.empty:
        zero_dte_snapshot = included[included['expiry'].eq(included['expiry'].min())].copy()
    zero_dte_by_strike = zero_dte_snapshot.groupby('strike').agg(
        open_interest=('open_interest', 'sum'),
        call_oi=('call_oi', 'sum'),
        put_oi=('put_oi', 'sum'),
    ).sort_index()

    call_wall = float(by_strike['call_gex'].idxmax())
    put_wall = float(by_strike['put_gex'].idxmax())

    positive_nodes = by_strike[by_strike['net_gex'] > 0]
    if positive_nodes.empty:
        long_gamma = call_wall
    else:
        long_gamma = float(positive_nodes['net_gex'].idxmax())

    cumulative = by_strike['net_gex'].cumsum()
    balance_score = (2.0 * cumulative - cumulative.iloc[-1]).abs()
    gex_pain = float(balance_score.idxmin())
    max_pain = _max_pain_strike(by_strike)

    front_expiry = included['expiry'].min()
    front_expiry_gex = float(included.loc[included['expiry'].eq(front_expiry), 'signed_gex'].sum())
    total_gex = float(included['signed_gex'].sum())
    total_gross_gex = float(included['gross_gex'].sum())

    local_window = by_strike[(by_strike.index >= spot_price - 50) & (by_strike.index <= spot_price + 50)]
    if len(local_window) < 3:
        local_window = by_strike.iloc[max(len(by_strike) // 2 - 2, 0): max(len(by_strike) // 2 + 3, 3)]
    if len(local_window) >= 2:
        gamma_slope = float(np.polyfit(local_window.index.to_numpy(dtype=float), local_window['net_gex'].to_numpy(dtype=float), 1)[0])
    else:
        gamma_slope = 0.0

    total_call = float(by_strike['call_gex'].sum())
    total_put = float(by_strike['put_gex'].sum())
    call_concentration = float(by_strike['call_gex'].max() / total_call) if total_call > 0 else 0.0
    put_concentration = float(by_strike['put_gex'].max() / total_put) if total_put > 0 else 0.0
    total_size_proxy = float(by_strike['size_proxy'].sum())
    strike_grid = np.diff(np.sort(by_strike.index.to_numpy(dtype=float)))
    strike_step = float(np.median(strike_grid)) if len(strike_grid) else 5.0
    expected_move = _expected_move(included, spot_price, as_of, session_close)

    by_strike['pin_risk_score'] = (
        np.minimum(by_strike['call_size_proxy'], by_strike['put_size_proxy']) /
        np.maximum(by_strike['size_proxy'], 1.0)
    )
    by_strike['gross_gex_share'] = by_strike['gross_gex'] / max(total_gross_gex, 1.0)
    by_strike['size_share'] = by_strike['size_proxy'] / max(total_size_proxy, 1.0)
    by_strike['distance_score'] = 1.0 / (
        1.0 + ((by_strike.index.to_numpy(dtype=float) - spot_price) / max(expected_move, strike_step)) ** 2
    )
    by_strike['attractor_score'] = (
        0.40 * by_strike['gross_gex_share'] +
        0.25 * by_strike['size_share'] +
        0.20 * by_strike['pin_risk_score'] +
        0.15 * by_strike['distance_score']
    )

    weighted_gamma_center = float(np.average(
        by_strike.index.to_numpy(dtype=float),
        weights=np.maximum(by_strike['attractor_score'].to_numpy(dtype=float), 1e-9),
    ))
    pin_risk_level = float(by_strike['pin_risk_score'].idxmax())
    pin_risk_score = float(by_strike['pin_risk_score'].max())
    (
        by_strike,
        weighted_gamma_magnet_price,
        strongest_magnet_strike,
        time_to_close_decay,
        gamma_rank_levels,
    ) = _compute_weighted_gamma_magnet(
        by_strike=by_strike,
        spot_price=spot_price,
        expected_move=expected_move,
        strike_step=strike_step,
        as_of=as_of,
        session_close=session_close,
    )
    by_strike, strongest_pin_strike, strongest_pin_score = _compute_pin_surface(
        by_strike=by_strike,
        zero_dte_by_strike=zero_dte_by_strike,
        spot_price=spot_price,
        expected_move=expected_move,
        strike_step=strike_step,
        as_of=as_of,
        session_close=session_close,
        intraday_vol_points=intraday_vol_points,
    )
    gamma_rank_levels = [
        GammaRankLevel(
            rank=level.rank,
            strike=level.strike,
            score=level.score,
            magnet_weight=level.magnet_weight,
            pin_score=float(by_strike.loc[level.strike, 'pin_score']),
            gross_gex=level.gross_gex,
            net_gex=level.net_gex,
            call_gex=level.call_gex,
            put_gex=level.put_gex,
            open_interest=float(by_strike.loc[level.strike, 'open_interest']),
            size_proxy=level.size_proxy,
            zero_dte_oi=float(by_strike.loc[level.strike, 'zero_dte_open_interest']),
            gamma_exposure_share=level.gamma_exposure_share,
            gamma_density=float(by_strike.loc[level.strike, 'gamma_density']),
            oi_concentration=level.oi_concentration,
            distance_decay=level.distance_decay,
            time_to_close_decay=level.time_to_close_decay,
            pin_risk_score=level.pin_risk_score,
        )
        for level in gamma_rank_levels
    ]
    market_regime = _market_regime(total_gex, total_gross_gex)

    if total_gex > 0 and put_wall <= spot_price <= call_wall:
        gamma_regime = 'positive-gamma pin'
    elif total_gex > 0:
        gamma_regime = 'positive-gamma'
    elif total_gex < 0 and market_regime == 'short_gamma':
        gamma_regime = 'negative-gamma'
    else:
        gamma_regime = 'mixed'

    return AnchorStructure(
        as_of=as_of,
        spot_price=float(spot_price),
        call_wall=call_wall,
        put_wall=put_wall,
        long_gamma=float(long_gamma),
        gex_pain=float(gex_pain),
        gamma_flip=_gamma_flip(by_strike['net_gex'], spot_price),
        expected_move=expected_move,
        total_gex=total_gex,
        total_gross_gex=total_gross_gex,
        front_expiry_gex=front_expiry_gex,
        gamma_slope=gamma_slope,
        call_concentration=call_concentration,
        put_concentration=put_concentration,
        gamma_regime=gamma_regime,
        market_regime=market_regime,
        weighted_gamma_center=weighted_gamma_center,
        weighted_gamma_magnet_price=weighted_gamma_magnet_price,
        strongest_magnet_strike=strongest_magnet_strike,
        strongest_pin_strike=strongest_pin_strike,
        strongest_pin_score=strongest_pin_score,
        max_pain=max_pain,
        pin_risk_level=pin_risk_level,
        pin_risk_score=pin_risk_score,
        time_to_close_decay=time_to_close_decay,
        expiry_bucket_gex={str(key): float(value) for key, value in expiry_bucket_gex.items()},
        gamma_rank_levels=gamma_rank_levels,
    )

import pandas as pd

from src.spx_anchor.types import AnchorForecast


def _fmt_price(value) -> str:
    if value is None or pd.isna(value):
        return 'n/a'
    return f"{float(value):,.1f}"


def _fmt_rank(value) -> str:
    if value is None or pd.isna(value):
        return 'n/a'
    return f"{float(value) * 100:.0f}%"


def _fmt_score(value) -> str:
    if value is None or pd.isna(value):
        return 'n/a'
    return f"{float(value):.3f}"


def render_markdown_summary(forecast: AnchorForecast) -> str:
    structure = forecast.structure
    ranks = structure.gamma_ranks or {}
    feature_snapshot = forecast.feature_snapshot or {}
    pin_trust_multiplier = feature_snapshot.get('pin_trust_multiplier')
    shock_score = feature_snapshot.get('shock_score')
    deanchor_score = feature_snapshot.get('deanchor_score')

    lines = [
        f"## SPX Close Anchor | {forecast.as_of.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Spot:** {_fmt_price(forecast.spot_price)} | **Target:** {_fmt_price(forecast.target_price)} | **Range:** {_fmt_price(forecast.forecast_low)} to {_fmt_price(forecast.forecast_high)} | **Confidence:** {forecast.confidence_score:.0%}",
        "",
        "**Structure**",
        f"- Market regime: {forecast.market_regime or structure.market_regime}",
        f"- Long gamma: {_fmt_price(structure.long_gamma)}",
        f"- Call wall: {_fmt_price(structure.call_wall)}",
        f"- Put wall: {_fmt_price(structure.put_wall)}",
        f"- GEX pain: {_fmt_price(structure.gex_pain)}",
        f"- Weighted gamma center: {_fmt_price(structure.weighted_gamma_center)}",
        f"- Weighted gamma magnet price: {_fmt_price(structure.weighted_gamma_magnet_price)}",
        f"- Strongest magnet strike: {_fmt_price(structure.strongest_magnet_strike)}",
        f"- Strongest pin strike: {_fmt_price(structure.strongest_pin_strike)}",
        f"- Strongest pin score: {_fmt_score(structure.strongest_pin_score)}",
        f"- Pin trust multiplier: {_fmt_score(pin_trust_multiplier)}" if pin_trust_multiplier is not None else "- Pin trust multiplier: n/a",
        f"- Shock score: {_fmt_score(shock_score)}" if shock_score is not None else "- Shock score: n/a",
        f"- De-anchor score: {_fmt_score(deanchor_score)}" if deanchor_score is not None else "- De-anchor score: n/a",
        f"- Max pain: {_fmt_price(structure.max_pain)}",
        f"- Pin risk level: {_fmt_price(structure.pin_risk_level)}",
        f"- Gamma regime: {structure.gamma_regime}",
        f"- Shock reprice: {'yes' if forecast.shock_reprice else 'no'}",
        "",
        "**Gamma Levels**",
    ]

    for level in structure.gamma_rank_levels[:5]:
        lines.append(
            f"- Gamma rank {level.rank}: {_fmt_price(level.strike)} | weight {_fmt_score(level.magnet_weight)} | pin {_fmt_score(level.pin_score)} | gamma {_fmt_score(level.gamma_exposure_share)} | oi {_fmt_score(level.oi_concentration)} | dist {_fmt_score(level.distance_decay)}"
        )

    lines.extend([
        "",
        "**Regime Stats**",
        f"- Total GEX: {_fmt_rank(ranks.get('total_gex_rank'))}",
        f"- Front expiry GEX: {_fmt_rank(ranks.get('front_expiry_gex_rank'))}",
        f"- Gamma slope: {_fmt_rank(ranks.get('gamma_slope_rank'))}",
        f"- Call concentration: {_fmt_rank(ranks.get('call_concentration_rank'))}",
        f"- Put concentration: {_fmt_rank(ranks.get('put_concentration_rank'))}",
    ])

    if forecast.baselines:
        lines.extend([
            "",
            "**Baselines**",
        ])
        for key in ['baseline_1', 'baseline_2', 'baseline_3']:
            baseline = forecast.baselines.get(key)
            if not baseline:
                continue
            lines.append(
                f"- {baseline['label']}: {_fmt_price(baseline['target_price'])} | weight {baseline['weight']:.0%}"
            )

    if forecast.drivers:
        lines.extend([
            "",
            "**Drivers**",
        ])
        lines.extend([f"- {driver}" for driver in forecast.drivers])

    if forecast.invalidation_conditions:
        lines.extend([
            "",
            "**Invalidation**",
        ])
        lines.extend([f"- {condition}" for condition in forecast.invalidation_conditions])

    return '\n'.join(lines)

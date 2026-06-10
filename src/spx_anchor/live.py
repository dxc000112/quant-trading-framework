from typing import Optional

import pandas as pd

from src.spx_anchor.baselines import build_baseline_forecast
from src.spx_anchor.config import SpxAnchorSettings
from src.spx_anchor.features import build_feature_snapshot
from src.spx_anchor.model import AnchorModelBundle, predict_anchor_forecast
from src.spx_anchor.reporting import render_markdown_summary
from src.spx_anchor.structure import latest_snapshot_as_of
from src.spx_anchor.versioned import build_versioned_forecast


def build_live_feature_frame(
    spot_bars: pd.DataFrame,
    option_snapshots: pd.DataFrame,
    settings: Optional[SpxAnchorSettings] = None,
    rank_history: Optional[pd.DataFrame] = None,
    futures_bars: Optional[pd.DataFrame] = None,
    event_flags: Optional[dict] = None,
):
    settings = settings or SpxAnchorSettings()
    if spot_bars is None or spot_bars.empty:
        raise ValueError("spot_bars must contain at least one row.")

    bars = spot_bars.sort_index().copy()
    as_of = pd.Timestamp(bars.index[-1])
    session_bars = bars[bars.index.normalize() == as_of.normalize()].copy()
    if session_bars.empty:
        raise ValueError("No current-session spot bars were available.")

    snapshot = latest_snapshot_as_of(
        options=option_snapshots,
        as_of=as_of,
        max_staleness_minutes=settings.max_snapshot_staleness_minutes,
    )
    if snapshot is None or snapshot.empty:
        raise ValueError("No fresh option snapshot was available for the live update.")

    prior = bars[bars.index < session_bars.index.min()]
    previous_close = float(prior['Close'].iloc[-1]) if not prior.empty else None

    feature_row, structure = build_feature_snapshot(
        session_bars=session_bars,
        option_snapshot=snapshot,
        settings=settings,
        previous_close=previous_close,
        rank_reference=rank_history,
        futures_bars=futures_bars,
        event_flags=event_flags,
    )
    feature_frame = pd.DataFrame([feature_row])
    return feature_frame, structure


def generate_baseline_forecast(
    spot_bars: pd.DataFrame,
    option_snapshots: pd.DataFrame,
    settings: Optional[SpxAnchorSettings] = None,
    rank_history: Optional[pd.DataFrame] = None,
    futures_bars: Optional[pd.DataFrame] = None,
    event_flags: Optional[dict] = None,
):
    feature_frame, structure = build_live_feature_frame(
        spot_bars=spot_bars,
        option_snapshots=option_snapshots,
        settings=settings,
        rank_history=rank_history,
        futures_bars=futures_bars,
        event_flags=event_flags,
    )
    return build_baseline_forecast(feature_frame.iloc[0], structure)


def generate_live_forecast(
    spot_bars: pd.DataFrame,
    option_snapshots: pd.DataFrame,
    model_bundle: AnchorModelBundle,
    settings: Optional[SpxAnchorSettings] = None,
    rank_history: Optional[pd.DataFrame] = None,
    futures_bars: Optional[pd.DataFrame] = None,
    event_flags: Optional[dict] = None,
):
    feature_frame, structure = build_live_feature_frame(
        spot_bars=spot_bars,
        option_snapshots=option_snapshots,
        settings=settings,
        rank_history=rank_history,
        futures_bars=futures_bars,
        event_flags=event_flags,
    )
    forecast = predict_anchor_forecast(model_bundle, feature_frame.iloc[0], structure)
    return forecast


def generate_live_markdown(
    spot_bars: pd.DataFrame,
    option_snapshots: pd.DataFrame,
    model_bundle: AnchorModelBundle,
    settings: Optional[SpxAnchorSettings] = None,
    rank_history: Optional[pd.DataFrame] = None,
    futures_bars: Optional[pd.DataFrame] = None,
    event_flags: Optional[dict] = None,
) -> str:
    forecast = generate_live_forecast(
        spot_bars=spot_bars,
        option_snapshots=option_snapshots,
        model_bundle=model_bundle,
        settings=settings,
        rank_history=rank_history,
        futures_bars=futures_bars,
        event_flags=event_flags,
    )
    return render_markdown_summary(forecast)


def generate_baseline_markdown(
    spot_bars: pd.DataFrame,
    option_snapshots: pd.DataFrame,
    settings: Optional[SpxAnchorSettings] = None,
    rank_history: Optional[pd.DataFrame] = None,
    futures_bars: Optional[pd.DataFrame] = None,
    event_flags: Optional[dict] = None,
) -> str:
    forecast = generate_baseline_forecast(
        spot_bars=spot_bars,
        option_snapshots=option_snapshots,
        settings=settings,
        rank_history=rank_history,
        futures_bars=futures_bars,
        event_flags=event_flags,
    )
    return render_markdown_summary(forecast)


def generate_versioned_forecast(
    spot_bars: pd.DataFrame,
    option_snapshots: pd.DataFrame,
    version: str = 'v3',
    settings: Optional[SpxAnchorSettings] = None,
    rank_history: Optional[pd.DataFrame] = None,
    futures_bars: Optional[pd.DataFrame] = None,
    event_flags: Optional[dict] = None,
):
    feature_frame, structure = build_live_feature_frame(
        spot_bars=spot_bars,
        option_snapshots=option_snapshots,
        settings=settings,
        rank_history=rank_history,
        futures_bars=futures_bars,
        event_flags=event_flags,
    )
    return build_versioned_forecast(feature_frame.iloc[0], structure, version=version)


def generate_versioned_markdown(
    spot_bars: pd.DataFrame,
    option_snapshots: pd.DataFrame,
    version: str = 'v3',
    settings: Optional[SpxAnchorSettings] = None,
    rank_history: Optional[pd.DataFrame] = None,
    futures_bars: Optional[pd.DataFrame] = None,
    event_flags: Optional[dict] = None,
) -> str:
    forecast = generate_versioned_forecast(
        spot_bars=spot_bars,
        option_snapshots=option_snapshots,
        version=version,
        settings=settings,
        rank_history=rank_history,
        futures_bars=futures_bars,
        event_flags=event_flags,
    )
    return render_markdown_summary(forecast)

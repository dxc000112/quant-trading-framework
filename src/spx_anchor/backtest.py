from typing import Optional

import numpy as np
import pandas as pd

from src.spx_anchor.baselines import predict_baseline_levels
from src.spx_anchor.config import SpxAnchorSettings
from src.spx_anchor.features import add_historical_rank_features, build_feature_snapshot
from src.spx_anchor.model import predict_anchor_levels, train_anchor_model
from src.spx_anchor.structure import latest_snapshot_as_of, normalize_option_snapshot
from src.spx_anchor.targets import label_future_distribution
from src.spx_anchor.versioned import SUPPORTED_VERSIONS, predict_version_levels


def _time_on_date(ts: pd.Timestamp, clock: str) -> pd.Timestamp:
    hour, minute = (int(part) for part in clock.split(':'))
    return ts.normalize() + pd.Timedelta(hours=hour, minutes=minute)


def _previous_close(spot_bars: pd.DataFrame, session_start: pd.Timestamp) -> Optional[float]:
    prior = spot_bars[spot_bars.index < session_start]
    if prior.empty:
        return None
    return float(pd.to_numeric(prior['Close'], errors='coerce').iloc[-1])


def _decision_times(session_bars: pd.DataFrame, settings: SpxAnchorSettings):
    if session_bars.empty:
        return []

    session_date = pd.Timestamp(session_bars.index[0]).normalize()
    first_time = session_date + pd.Timedelta(
        hours=int(settings.anchor_start_time.split(':')[0]),
        minutes=int(settings.anchor_start_time.split(':')[1]),
    )
    valid = session_bars[session_bars.index >= first_time]
    if valid.empty:
        return []

    decision_times = []
    for timestamp in valid.index:
        minute_offset = int((timestamp - first_time).total_seconds() // 60)
        if minute_offset % settings.update_frequency_minutes == 0:
            decision_times.append(timestamp)
    return decision_times


def build_snapshot_panel(
    spot_bars: pd.DataFrame,
    option_snapshots: pd.DataFrame,
    settings: Optional[SpxAnchorSettings] = None,
) -> pd.DataFrame:
    settings = settings or SpxAnchorSettings()
    if spot_bars is None or spot_bars.empty:
        raise ValueError("spot_bars must contain at least one row.")

    bars = spot_bars.sort_index().copy()
    options = normalize_option_snapshot(option_snapshots)
    records = []

    grouped = bars.groupby(bars.index.normalize())
    for session_date, day_frame in grouped:
        day_frame = day_frame.sort_index()
        session_open = _time_on_date(pd.Timestamp(session_date), settings.session_open)
        session_close = _time_on_date(pd.Timestamp(session_date), settings.session_close)
        session_bars = day_frame[(day_frame.index >= session_open) & (day_frame.index <= session_close)]
        if len(session_bars) < 35:
            continue

        previous_close = _previous_close(bars, session_open)

        for as_of in _decision_times(session_bars, settings):
            history = session_bars[session_bars.index <= as_of]
            future = session_bars[session_bars.index >= as_of]
            if len(history) < 10 or len(future) < 5:
                continue

            snapshot = latest_snapshot_as_of(
                options=options,
                as_of=as_of,
                max_staleness_minutes=settings.max_snapshot_staleness_minutes,
            )
            if snapshot is None or snapshot.empty:
                continue

            feature_row, _ = build_feature_snapshot(
                session_bars=history,
                option_snapshot=snapshot,
                settings=settings,
                previous_close=previous_close,
            )
            labels = label_future_distribution(
                future_bars=future,
                price_bin_size=settings.price_bin_size,
                lower_quantile=settings.lower_interval_quantile,
                upper_quantile=settings.upper_interval_quantile,
            )

            record = feature_row.to_dict()
            record.update({
                'label_target_price': labels['target_price'],
                'label_lower_bound': labels['lower_bound'],
                'label_upper_bound': labels['upper_bound'],
                'future_close': labels['future_close'],
                'future_vwap': labels['future_vwap'],
            })
            records.append(record)

    panel = pd.DataFrame(records)
    if panel.empty:
        raise ValueError("No valid snapshot rows were built. Check data coverage and timestamps.")

    panel['session_date'] = pd.to_datetime(panel['session_date'])
    panel['as_of'] = pd.to_datetime(panel['as_of'])
    panel = add_historical_rank_features(
        panel,
        lookback_rows=settings.lookback_rows_for_ranks,
    )
    return panel.sort_values(['session_date', 'as_of']).reset_index(drop=True)


def summarize_backtest(results: pd.DataFrame):
    if results is None or results.empty:
        raise ValueError("results must contain at least one prediction row.")

    target_error = (results['pred_target_price'] - results['label_target_price']).abs()
    spot_error = (results['spot_price'] - results['label_target_price']).abs()
    pain_error = (results['gex_pain'] - results['label_target_price']).abs()
    interval_width = results['pred_upper_bound'] - results['pred_lower_bound']
    covered = (
        (results['label_target_price'] >= results['pred_lower_bound']) &
        (results['label_target_price'] <= results['pred_upper_bound'])
    )

    return {
        'rows': int(len(results)),
        'sessions': int(results['session_date'].nunique()),
        'target_mae': float(target_error.mean()),
        'target_rmse': float(np.sqrt(np.mean((results['pred_target_price'] - results['label_target_price']) ** 2))),
        'avg_interval_width': float(interval_width.mean()),
        'interval_coverage': float(covered.mean()),
        'hit_rate_5pts': float((target_error <= 5.0).mean()),
        'hit_rate_10pts': float((target_error <= 10.0).mean()),
        'spot_baseline_mae': float(spot_error.mean()),
        'gex_pain_baseline_mae': float(pain_error.mean()),
    }


def run_baseline_backtest(
    snapshot_panel: pd.DataFrame,
    settings: Optional[SpxAnchorSettings] = None,
):
    settings = settings or SpxAnchorSettings()
    panel = snapshot_panel.sort_values(['session_date', 'as_of']).reset_index(drop=True).copy()
    predictions = predict_baseline_levels(panel)
    results = pd.concat([panel.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    summary = summarize_backtest(results)
    summary['engine'] = 'baseline-strike-concentration'
    return results, summary


def run_walk_forward_backtest(
    snapshot_panel: pd.DataFrame,
    settings: Optional[SpxAnchorSettings] = None,
):
    settings = settings or SpxAnchorSettings()
    panel = snapshot_panel.sort_values(['session_date', 'as_of']).reset_index(drop=True).copy()
    sessions = panel['session_date'].drop_duplicates().sort_values().tolist()

    # Reserve the final holdout_fraction of sessions as a locked test set.
    # Walk-forward development only runs on dev_sessions.
    holdout_start_idx = int(len(sessions) * (1 - settings.holdout_fraction))
    dev_sessions = sessions[:holdout_start_idx]
    holdout_sessions = sessions[holdout_start_idx:]

    if len(dev_sessions) <= settings.min_train_sessions:
        raise ValueError(
            f"Not enough dev sessions ({len(dev_sessions)}) for walk-forward backtest "
            f"(min_train_sessions={settings.min_train_sessions}). "
            f"Add more history or reduce holdout_fraction."
        )

    predictions = []
    model_bundle = None
    last_fit_idx = -1

    for idx in range(settings.min_train_sessions, len(dev_sessions)):
        if model_bundle is None or (idx - last_fit_idx) >= settings.retrain_every_n_sessions:
            train_sessions = dev_sessions[:idx]
            train_panel = panel[panel['session_date'].isin(train_sessions)].copy()
            model_bundle = train_anchor_model(train_panel)
            last_fit_idx = idx

        test_session = dev_sessions[idx]
        test_panel = panel[panel['session_date'].eq(test_session)].copy()
        if test_panel.empty:
            continue

        pred = predict_anchor_levels(model_bundle, test_panel)
        merged = pd.concat([test_panel.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        predictions.append(merged)

    if not predictions:
        raise ValueError("Walk-forward backtest did not produce any predictions.")

    results = pd.concat(predictions, ignore_index=True)
    summary = summarize_backtest(results)
    summary['dev_sessions'] = len(dev_sessions)
    summary['holdout_sessions_locked'] = len(holdout_sessions)
    summary['holdout_start_date'] = str(holdout_sessions[0].date()) if holdout_sessions else None
    return results, summary, model_bundle


def run_holdout_evaluation(
    snapshot_panel: pd.DataFrame,
    model_bundle,
    settings: Optional[SpxAnchorSettings] = None,
) -> tuple:
    """Final evaluation on the locked holdout set. Call this only once, after all tuning is done."""
    settings = settings or SpxAnchorSettings()
    panel = snapshot_panel.sort_values(['session_date', 'as_of']).reset_index(drop=True).copy()
    sessions = panel['session_date'].drop_duplicates().sort_values().tolist()

    holdout_start_idx = int(len(sessions) * (1 - settings.holdout_fraction))
    holdout_sessions = sessions[holdout_start_idx:]

    if not holdout_sessions:
        raise ValueError("No holdout sessions available. Check holdout_fraction setting.")

    holdout_panel = panel[panel['session_date'].isin(holdout_sessions)].copy()
    pred = predict_anchor_levels(model_bundle, holdout_panel)
    results = pd.concat([holdout_panel.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    summary = summarize_backtest(results)
    summary['engine'] = 'holdout-final'
    summary['holdout_sessions'] = len(holdout_sessions)
    summary['holdout_start_date'] = str(holdout_sessions[0].date())
    return results, summary


def run_version_backtest(
    snapshot_panel: pd.DataFrame,
    version: str = 'v1',
    settings: Optional[SpxAnchorSettings] = None,
):
    settings = settings or SpxAnchorSettings()
    version = str(version).lower()
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported version: {version}")

    panel = snapshot_panel.sort_values(['session_date', 'as_of']).reset_index(drop=True).copy()
    predictions = predict_version_levels(panel, version=version)
    results = pd.concat([panel.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    summary = summarize_backtest(results)
    summary.update({
        'engine': f'spx-anchor-{version}',
        'version': version,
        'avg_confidence_score': float(results['confidence_score'].mean()) if 'confidence_score' in results.columns else np.nan,
    })
    return results, summary


def run_version_comparison_backtest(
    snapshot_panel: pd.DataFrame,
    settings: Optional[SpxAnchorSettings] = None,
    versions=None,
):
    settings = settings or SpxAnchorSettings()
    versions = [str(version).lower() for version in (versions or SUPPORTED_VERSIONS)]

    summaries = []
    results_map = {}
    previous_summary = None
    for version in versions:
        results, summary = run_version_backtest(snapshot_panel, version=version, settings=settings)
        comparison_summary = summary.copy()
        if previous_summary is not None:
            comparison_summary['delta_target_mae_vs_prev'] = float(summary['target_mae'] - previous_summary['target_mae'])
            comparison_summary['delta_interval_coverage_vs_prev'] = float(summary['interval_coverage'] - previous_summary['interval_coverage'])
            comparison_summary['delta_hit_rate_10pts_vs_prev'] = float(summary['hit_rate_10pts'] - previous_summary['hit_rate_10pts'])
        else:
            comparison_summary['delta_target_mae_vs_prev'] = np.nan
            comparison_summary['delta_interval_coverage_vs_prev'] = np.nan
            comparison_summary['delta_hit_rate_10pts_vs_prev'] = np.nan

        summaries.append(comparison_summary)
        results_map[version] = results
        previous_summary = summary

    return results_map, pd.DataFrame(summaries)

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.spx_anchor.backtest import build_snapshot_panel, run_version_backtest, run_version_comparison_backtest
from src.spx_anchor.config import SpxAnchorSettings
from src.spx_anchor.live import generate_versioned_forecast
from src.spx_anchor.reporting import render_markdown_summary
from src.spx_anchor.synthetic import make_synthetic_spx_dataset
from src.spx_anchor.versioned import SUPPORTED_VERSIONS


def _to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ''
    headers = [str(column) for column in frame.columns]
    rows = [headers, ['---'] * len(headers)]
    for _, row in frame.iterrows():
        rows.append([str(value) for value in row.tolist()])
    return '\n'.join(['| ' + ' | '.join(row) + ' |' for row in rows])


def _live_demo_forecast(panel: pd.DataFrame, spot_bars: pd.DataFrame, option_snapshots: pd.DataFrame, version: str, settings: SpxAnchorSettings):
    sessions = panel['session_date'].drop_duplicates().sort_values()
    train_panel = panel[panel['session_date'].isin(sessions[:-1])].copy()
    last_session = sessions.iloc[-1]
    live_bars = spot_bars[spot_bars.index.normalize() == last_session].copy()
    live_bars = live_bars[live_bars.index <= live_bars.index[65]]
    live_options = option_snapshots[
        (pd.to_datetime(option_snapshots['as_of']).dt.normalize() == last_session) &
        (pd.to_datetime(option_snapshots['as_of']) <= live_bars.index[-1])
    ].copy()

    return generate_versioned_forecast(
        spot_bars=pd.concat([
            spot_bars[spot_bars.index.normalize() < last_session],
            live_bars,
        ]),
        option_snapshots=live_options,
        version=version,
        settings=settings,
        rank_history=train_panel,
        event_flags={'opex': True} if version == 'v3' else None,
    )


def generate_versioned_backtest_artifacts(
    output_dir: str = 'backtest_outputs/spx_anchor_versions',
    settings: Optional[SpxAnchorSettings] = None,
    sessions: int = 14,
):
    settings = settings or SpxAnchorSettings(
        min_train_sessions=6,
        retrain_every_n_sessions=2,
        lookback_rows_for_ranks=160,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    spot_bars, option_snapshots = make_synthetic_spx_dataset(sessions=sessions)
    panel = build_snapshot_panel(spot_bars, option_snapshots, settings=settings)
    results_map, comparison = run_version_comparison_backtest(panel, settings=settings)

    comparison_export = comparison.copy()
    if not comparison_export.empty:
        for column in comparison_export.columns:
            if comparison_export[column].dtype.kind in {'f', 'i'}:
                comparison_export[column] = comparison_export[column].map(
                    lambda value: round(float(value), 6) if pd.notna(value) else value
                )
    comparison_export.to_csv(output_path / 'comparison_summary.csv', index=False)
    (output_path / 'comparison_summary.json').write_text(
        comparison.to_json(orient='records', indent=2, date_format='iso'),
        encoding='utf-8',
    )

    version_reports = []
    for version in SUPPORTED_VERSIONS:
        results = results_map[version]
        _, summary = run_version_backtest(panel, version=version, settings=settings)
        results.to_csv(output_path / f'{version}_backtest.csv', index=False)
        (output_path / f'{version}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

        forecast = _live_demo_forecast(panel, spot_bars, option_snapshots, version=version, settings=settings)
        report_lines = [
            f"# SPX Anchor {version.upper()} Backtest Report",
            "",
            "Synthetic-dataset report generated from the versioned research scaffold. Replace the synthetic input with real SPX/SPXW snapshots to obtain production-grade metrics.",
            "",
            "## Summary",
            "",
        ]
        summary_table = pd.DataFrame([summary])[[
            'version', 'target_mae', 'target_rmse', 'interval_coverage',
            'avg_interval_width', 'hit_rate_5pts', 'hit_rate_10pts', 'avg_confidence_score',
        ]].copy()
        report_lines.append(_to_markdown_table(summary_table))
        report_lines.extend([
            "",
            "## Live Example",
            "",
            render_markdown_summary(forecast),
            "",
            "## Latest Backtest Rows",
            "",
            _to_markdown_table(
                results[[
                    'as_of', 'label_target_price', 'pred_target_price',
                    'pred_lower_bound', 'pred_upper_bound', 'confidence_score',
                ]].tail(10).assign(
                    as_of=lambda frame: frame['as_of'].astype(str)
                )
            ),
        ])

        report_text = '\n'.join(report_lines)
        (output_path / f'{version}_report.md').write_text(report_text, encoding='utf-8')
        version_reports.append({'version': version, **summary})

    comparison_lines = [
        "# SPX Anchor Version Comparison",
        "",
        "This comparison uses the same synthetic session set for `V1`, `V2`, and `V3`. The goal is to verify that the version stack is executable and comparable before plugging in real vendor data.",
        "",
        "## Comparison Table",
        "",
        _to_markdown_table(
            comparison[[
                'version', 'target_mae', 'target_rmse', 'interval_coverage',
                'avg_interval_width', 'hit_rate_5pts', 'hit_rate_10pts',
                'avg_confidence_score', 'delta_target_mae_vs_prev',
                'delta_interval_coverage_vs_prev', 'delta_hit_rate_10pts_vs_prev',
            ]]
        ),
        "",
        "## Interpretation Notes",
        "",
        "- `V1` is the simplest wall and max-pain anchor.",
        "- `V2` adds strike concentration, 0DTE weighting, time decay, and intraday mean-reversion features.",
        "- `V3` adds regime-aware target selection, confidence scoring, and invalidation logic.",
        "- Deltas are computed versus the immediately previous version.",
    ]
    (output_path / 'version_comparison_report.md').write_text('\n'.join(comparison_lines), encoding='utf-8')

    return {
        'panel_rows': int(len(panel)),
        'output_dir': str(output_path),
        'comparison': comparison,
        'version_reports': pd.DataFrame(version_reports),
    }


def main():
    payload = generate_versioned_backtest_artifacts()
    print(f"Saved reports to {payload['output_dir']}")
    print(payload['comparison'].to_string(index=False))


if __name__ == '__main__':
    main()

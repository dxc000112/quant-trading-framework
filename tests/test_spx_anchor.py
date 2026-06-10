import unittest

import numpy as np
import pandas as pd

from src.spx_anchor.backtest import build_snapshot_panel, run_baseline_backtest, run_walk_forward_backtest
from src.spx_anchor.config import SpxAnchorSettings
from src.spx_anchor.live import generate_baseline_forecast, generate_live_forecast
from src.spx_anchor.model import train_anchor_model
from src.spx_anchor.reporting import render_markdown_summary


def make_synthetic_spx_dataset(seed: int = 7, sessions: int = 12):
    rng = np.random.default_rng(seed)
    all_bars = []
    all_options = []
    base_price = 5200.0

    for day_idx in range(sessions):
        session_date = pd.Timestamp('2024-01-02') + pd.Timedelta(days=day_idx)
        minute_index = pd.date_range(
            session_date + pd.Timedelta(hours=9, minutes=30),
            session_date + pd.Timedelta(hours=16),
            freq='1min',
        )
        if len(minute_index) > 390:
            minute_index = minute_index[:390]

        latent = rng.normal(0, 1)
        target_price = base_price + latent * 12 + day_idx * 1.5
        call_wall = round((target_price + 20) / 5) * 5
        put_wall = round((target_price - 20) / 5) * 5
        long_gamma = round(target_price / 5) * 5

        prices = []
        current = target_price - latent * 6
        for idx, timestamp in enumerate(minute_index):
            if idx < 30:
                drift = 0.18 * latent
            else:
                drift = 0.10 * (target_price - current)
            noise = rng.normal(0, 1.1)
            current = current + drift + noise
            prices.append(current)

        close = pd.Series(prices, index=minute_index)
        open_ = close.shift(1).fillna(close.iloc[0] - rng.normal(0, 1.5))
        high = pd.concat([open_, close], axis=1).max(axis=1) + rng.uniform(0.2, 1.8, size=len(close))
        low = pd.concat([open_, close], axis=1).min(axis=1) - rng.uniform(0.2, 1.8, size=len(close))
        volume = rng.integers(8_000, 24_000, size=len(close)).astype(float)
        typical = (high + low + close) / 3.0
        vwap = (typical * volume).cumsum() / np.maximum(volume.cumsum(), 1.0)

        bars = pd.DataFrame({
            'Open': open_.values,
            'High': high.values,
            'Low': low.values,
            'Close': close.values,
            'Volume': volume,
            'VWAP': vwap.values,
        }, index=minute_index)
        all_bars.append(bars)

        snapshot_times = pd.date_range(
            session_date + pd.Timedelta(hours=10),
            session_date + pd.Timedelta(hours=15, minutes=55),
            freq='5min',
        )
        strikes = np.arange(long_gamma - 40, long_gamma + 45, 5)

        for snapshot_time in snapshot_times:
            for strike in strikes:
                call_oi = 80 + 900 * np.exp(-((strike - call_wall) / 7.5) ** 2) + 550 * np.exp(-((strike - long_gamma) / 10.0) ** 2)
                put_oi = 80 + 900 * np.exp(-((strike - put_wall) / 7.5) ** 2)
                call_gamma = 0.010 + 0.020 * np.exp(-((strike - long_gamma) / 12.0) ** 2)
                put_gamma = 0.009 + 0.018 * np.exp(-((strike - put_wall) / 11.0) ** 2)
                iv = 0.14 + 0.01 * abs(latent)

                all_options.append({
                    'as_of': snapshot_time,
                    'expiry': session_date,
                    'option_type': 'C',
                    'strike': float(strike),
                    'gamma': float(call_gamma),
                    'open_interest': float(call_oi),
                    'volume': float(max(call_oi * 0.08 + rng.normal(0, 8), 1)),
                    'iv': float(iv),
                })
                all_options.append({
                    'as_of': snapshot_time,
                    'expiry': session_date,
                    'option_type': 'P',
                    'strike': float(strike),
                    'gamma': float(put_gamma),
                    'open_interest': float(put_oi),
                    'volume': float(max(put_oi * 0.08 + rng.normal(0, 8), 1)),
                    'iv': float(iv),
                })

    return pd.concat(all_bars).sort_index(), pd.DataFrame(all_options)


class SpxAnchorTests(unittest.TestCase):
    def test_snapshot_panel_contains_structure_and_labels(self):
        spot_bars, option_snapshots = make_synthetic_spx_dataset()
        settings = SpxAnchorSettings(
            price_bin_size=5.0,
            min_train_sessions=6,
            retrain_every_n_sessions=2,
            lookback_rows_for_ranks=120,
        )

        panel = build_snapshot_panel(spot_bars, option_snapshots, settings=settings)

        self.assertFalse(panel.empty)
        self.assertIn('label_target_price', panel.columns)
        self.assertIn('call_wall', panel.columns)
        self.assertIn('gex_pain', panel.columns)
        self.assertIn('total_gex_rank', panel.columns)

    def test_walk_forward_backtest_and_live_card_render(self):
        spot_bars, option_snapshots = make_synthetic_spx_dataset()
        settings = SpxAnchorSettings(
            price_bin_size=5.0,
            min_train_sessions=6,
            retrain_every_n_sessions=2,
            lookback_rows_for_ranks=120,
        )

        panel = build_snapshot_panel(spot_bars, option_snapshots, settings=settings)
        results, summary, _ = run_walk_forward_backtest(panel, settings=settings)

        self.assertFalse(results.empty)
        self.assertIn('target_mae', summary)
        self.assertIn('interval_coverage', summary)

        sessions = panel['session_date'].drop_duplicates().sort_values()
        train_panel = panel[panel['session_date'].isin(sessions[:-1])].copy()
        rank_history = train_panel.copy()
        model_bundle = train_anchor_model(train_panel)

        last_session = sessions.iloc[-1]
        live_bars = spot_bars[spot_bars.index.normalize() == last_session].copy()
        live_bars = live_bars[live_bars.index <= live_bars.index[65]]
        live_options = option_snapshots[
            (pd.to_datetime(option_snapshots['as_of']).dt.normalize() == last_session) &
            (pd.to_datetime(option_snapshots['as_of']) <= live_bars.index[-1])
        ].copy()

        forecast = generate_live_forecast(
            spot_bars=pd.concat([
                spot_bars[spot_bars.index.normalize() < last_session],
                live_bars,
            ]),
            option_snapshots=live_options,
            model_bundle=model_bundle,
            settings=settings,
            rank_history=rank_history,
        )
        card = render_markdown_summary(forecast)

        self.assertIn('SPX Close Anchor', card)
        self.assertIn('Call wall', card)
        self.assertGreater(forecast.upper_bound, forecast.lower_bound)

    def test_baseline_backtest_and_output_fields(self):
        spot_bars, option_snapshots = make_synthetic_spx_dataset()
        settings = SpxAnchorSettings(
            price_bin_size=5.0,
            min_train_sessions=6,
            retrain_every_n_sessions=2,
            lookback_rows_for_ranks=120,
        )

        panel = build_snapshot_panel(spot_bars, option_snapshots, settings=settings)
        results, summary = run_baseline_backtest(panel, settings=settings)

        self.assertFalse(results.empty)
        self.assertEqual(summary['engine'], 'baseline-strike-concentration')
        self.assertIn('baseline_1_target', results.columns)
        self.assertIn('baseline_2_target', results.columns)
        self.assertIn('baseline_3_target', results.columns)
        self.assertIn('confidence_score', results.columns)

        sessions = panel['session_date'].drop_duplicates().sort_values()
        train_panel = panel[panel['session_date'].isin(sessions[:-1])].copy()
        last_session = sessions.iloc[-1]
        live_bars = spot_bars[spot_bars.index.normalize() == last_session].copy()
        live_bars = live_bars[live_bars.index <= live_bars.index[65]]
        live_options = option_snapshots[
            (pd.to_datetime(option_snapshots['as_of']).dt.normalize() == last_session) &
            (pd.to_datetime(option_snapshots['as_of']) <= live_bars.index[-1])
        ].copy()

        forecast = generate_baseline_forecast(
            spot_bars=pd.concat([
                spot_bars[spot_bars.index.normalize() < last_session],
                live_bars,
            ]),
            option_snapshots=live_options,
            settings=settings,
            rank_history=train_panel,
            event_flags={'opex': True},
        )
        card = render_markdown_summary(forecast)
        record = forecast.to_record()

        self.assertIn('Gamma rank 1', card)
        self.assertIn('Invalidation', card)
        self.assertEqual(record['market_regime'], forecast.market_regime)
        self.assertIn('gamma_rank_1', record)
        self.assertIn('confidence_score', record)


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import pandas as pd

from src.spx_anchor.backtest import build_snapshot_panel, run_version_backtest, run_version_comparison_backtest
from src.spx_anchor.config import SpxAnchorSettings
from src.spx_anchor.features import compute_pin_trust_features
from src.spx_anchor.live import generate_versioned_forecast
from src.spx_anchor.reporting import render_markdown_summary
from src.spx_anchor.synthetic import make_synthetic_spx_dataset
from src.spx_anchor.versioned import predict_version_row


class SpxAnchorVersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = SpxAnchorSettings(
            price_bin_size=5.0,
            min_train_sessions=6,
            retrain_every_n_sessions=2,
            lookback_rows_for_ranks=140,
        )
        cls.spot_bars, cls.option_snapshots = make_synthetic_spx_dataset(sessions=10)
        cls.panel = build_snapshot_panel(cls.spot_bars, cls.option_snapshots, settings=cls.settings)
        cls.sessions = cls.panel['session_date'].drop_duplicates().sort_values()
        cls.train_panel = cls.panel[cls.panel['session_date'].isin(cls.sessions[:-1])].copy()
        cls.last_session = cls.sessions.iloc[-1]
        cls.live_bars = cls.spot_bars[cls.spot_bars.index.normalize() == cls.last_session].copy()
        cls.live_bars = cls.live_bars[cls.live_bars.index <= cls.live_bars.index[65]]
        cls.live_options = cls.option_snapshots[
            (pd.to_datetime(cls.option_snapshots['as_of']).dt.normalize() == cls.last_session) &
            (pd.to_datetime(cls.option_snapshots['as_of']) <= cls.live_bars.index[-1])
        ].copy()

    def _forecast_for(self, version: str):
        return generate_versioned_forecast(
            spot_bars=pd.concat([
                self.spot_bars[self.spot_bars.index.normalize() < self.last_session],
                self.live_bars,
            ]),
            option_snapshots=self.live_options,
            version=version,
            settings=self.settings,
            rank_history=self.train_panel,
            event_flags={'opex': True, 'headline_shock_flag': version == 'v3'},
        )

    def test_version_1_backtest_and_live_forecast(self):
        results, summary = run_version_backtest(self.panel, version='v1', settings=self.settings)
        forecast = self._forecast_for('v1')

        self.assertFalse(results.empty)
        self.assertEqual(summary['version'], 'v1')
        self.assertIn('target_price', results.columns)
        self.assertIn('forecast_low', results.columns)
        self.assertIn('forecast_high', results.columns)
        self.assertGreater(forecast.forecast_high, forecast.forecast_low)

    def test_version_2_backtest_and_live_forecast(self):
        results, summary = run_version_backtest(self.panel, version='v2', settings=self.settings)
        forecast = self._forecast_for('v2')
        card = render_markdown_summary(forecast)

        self.assertFalse(results.empty)
        self.assertEqual(summary['version'], 'v2')
        self.assertIn('zero_dte_share', results.columns)
        self.assertIn('target_price', results.columns)
        self.assertIn('weighted_gamma_magnet_price', results.columns)
        self.assertIn('strongest_magnet_strike', results.columns)
        self.assertIn('Gamma Levels', card)
        self.assertGreater(forecast.confidence_score, 0.0)
        self.assertAlmostEqual(
            forecast.structure.strongest_magnet_strike,
            forecast.structure.gamma_rank_levels[0].strike,
            places=6,
        )

    def test_version_3_backtest_and_live_forecast(self):
        results, summary = run_version_backtest(self.panel, version='v3', settings=self.settings)
        forecast = self._forecast_for('v3')
        record = forecast.to_record()

        self.assertFalse(results.empty)
        self.assertEqual(summary['version'], 'v3')
        self.assertIn('version_regime', results.columns)
        self.assertIn(forecast.market_regime, {'long_gamma', 'short_gamma', 'event_shock', 'trend_day', 'pin_day'})
        self.assertTrue(forecast.invalidation_conditions)
        self.assertIn('confidence_score', record)
        self.assertIn('pin_trust_multiplier', record)
        self.assertIn('shock_score', record)

    def test_weighted_gamma_magnet_outputs(self):
        forecast = self._forecast_for('v3')
        structure = forecast.structure
        top_levels = structure.gamma_rank_levels[:5]

        self.assertGreaterEqual(len(top_levels), 5)
        self.assertAlmostEqual(structure.strongest_magnet_strike, top_levels[0].strike, places=6)
        self.assertGreater(top_levels[0].magnet_weight, 0.0)
        self.assertGreater(top_levels[0].pin_score, 0.0)
        self.assertGreater(top_levels[0].gamma_exposure_share, 0.0)
        self.assertGreater(top_levels[0].oi_concentration, 0.0)
        self.assertGreater(top_levels[0].distance_decay, 0.0)
        self.assertGreater(top_levels[0].gamma_density, 0.0)
        self.assertGreaterEqual(structure.strongest_pin_score, 0.0)
        self.assertLessEqual(structure.strongest_pin_score, 1.0)
        self.assertGreaterEqual(structure.time_to_close_decay, 0.0)
        self.assertLessEqual(structure.time_to_close_decay, 1.0)
        strikes = [level.strike for level in top_levels]
        self.assertGreaterEqual(structure.weighted_gamma_magnet_price, min(strikes) - 25.0)
        self.assertLessEqual(structure.weighted_gamma_magnet_price, max(strikes) + 25.0)
        self.assertIn('weighted_gamma_magnet_price', record := forecast.to_record())
        self.assertIn('strongest_magnet_strike', record)
        self.assertIn('strongest_pin_strike', record)
        self.assertIn('strongest_pin_score', record)

    def test_pin_trust_filter_opens_low_then_recovers(self):
        structure = self._forecast_for('v2').structure
        opening = compute_pin_trust_features(
            price_features={
                'spot_price': structure.spot_price + 12.0,
                'realized_vol_5m': 0.020,
                'realized_vol_30m': 0.010,
                'vwap_distance': 18.0,
                'ib_width': 24.0,
                'minutes_from_open': 20.0,
            },
            structure=structure,
            event_flags={'cpi': True},
        )
        recovery = compute_pin_trust_features(
            price_features={
                'spot_price': structure.strongest_pin_strike,
                'realized_vol_5m': 0.009,
                'realized_vol_30m': 0.010,
                'vwap_distance': 2.0,
                'ib_width': 24.0,
                'minutes_from_open': 150.0,
            },
            structure=structure,
            event_flags={},
        )

        self.assertEqual(opening['shock_state'], 'opening_uncertain')
        self.assertLess(opening['pin_trust_multiplier'], recovery['pin_trust_multiplier'])
        self.assertGreater(recovery['vol_convergence_score'], opening['vol_convergence_score'])
        self.assertLessEqual(opening['pin_trust_multiplier'], opening['opening_trust_cap'])

    def test_version_3_trend_day_target_stays_inside_structure(self):
        row = self.panel.iloc[-1].copy()
        row['shock_reprice'] = 0.0
        row['shock_state'] = 'normal'
        row['shock_score'] = 0.22
        row['pin_trust_multiplier'] = 0.18
        row['strongest_pin_score'] = 0.18
        row['ib_breakout'] = 1.0
        row['drive_efficiency'] = 0.45
        row['last_15m_return'] = 0.008
        row['vwap_distance'] = 55.0
        row['spot_es_divergence'] = 0.002

        payload = predict_version_row(row, version='v3')

        rank_strikes = []
        rank_scores = []
        for idx in range(1, 6):
            strike = row.get(f'gamma_rank_{idx}_strike', np.nan)
            score = row.get(f'gamma_rank_{idx}_score', np.nan)
            if pd.notna(strike) and pd.notna(score):
                rank_strikes.append(float(strike))
                rank_scores.append(max(float(score), 1e-9))
        rank_target = float(np.average(rank_strikes, weights=rank_scores))
        nearest_wall = float(
            row['put_wall']
            if abs(float(row['spot_price']) - float(row['put_wall'])) <= abs(float(row['spot_price']) - float(row['call_wall']))
            else row['call_wall']
        )
        structural_candidates = [
            float(row['weighted_gamma_magnet_price']),
            float(row['strongest_magnet_strike']),
            float(row['weighted_gamma_center']),
            rank_target,
            float(row['strongest_pin_strike']),
            float(row['max_pain']),
            nearest_wall,
        ]

        self.assertEqual(payload['version_regime'], 'trend_day')
        self.assertGreaterEqual(payload['target_price'], min(structural_candidates) - 6.0)
        self.assertLessEqual(payload['target_price'], max(structural_candidates) + 6.0)

    def test_version_comparison_backtest(self):
        results_map, comparison = run_version_comparison_backtest(self.panel, settings=self.settings)

        self.assertEqual(set(results_map.keys()), {'v1', 'v2', 'v3'})
        self.assertEqual(set(comparison['version']), {'v1', 'v2', 'v3'})
        self.assertIn('delta_target_mae_vs_prev', comparison.columns)
        self.assertEqual(len(comparison), 3)


if __name__ == '__main__':
    unittest.main()

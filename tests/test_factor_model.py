import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.backtests.backtest_factor_model import run_oos_backtest
from src.factor_library import generate_factor_library
from src.factor_selection_model import (
    build_focus_snapshot,
    pick_one_stock,
    pick_top_stocks,
    score_latest_cross_section,
    train_factor_selection_model,
)
from src.score_factor_model import resolve_model_path


def make_ohlcv(seed: int, periods: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-01', periods=periods, freq='D')

    drift = 0.0008 + (seed * 0.00005)
    shocks = rng.normal(drift, 0.018, size=periods)
    close = 100 * np.exp(np.cumsum(shocks))
    open_ = close * (1 + rng.normal(0, 0.003, size=periods))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.015, size=periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.015, size=periods))
    volume = rng.integers(800_000, 5_000_000, size=periods).astype(float)

    return pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)


class FactorModelTests(unittest.TestCase):
    def test_factor_library_generates_at_least_3000_columns(self):
        factors = generate_factor_library(make_ohlcv(seed=1))
        self.assertGreaterEqual(factors.shape[1], 3000)

    def test_training_and_scoring_pipeline_runs_on_synthetic_data(self):
        price_map = {f'TICK{i}': make_ohlcv(seed=i) for i in range(6)}

        model_bundle = train_factor_selection_model(
            price_map,
            horizon=5,
            min_factor_count=3000,
            top_factor_count=64,
            save_path=None,
        )

        self.assertGreaterEqual(len(model_bundle['selected_features']), 64)
        self.assertIn('mean_rank_ic', model_bundle['metrics'])

        scores = score_latest_cross_section(model_bundle, price_map)
        self.assertFalse(scores.empty)
        self.assertIn('score', scores.columns)
        self.assertEqual(scores.index.name, 'ticker')
        self.assertIn('market', scores.columns)
        self.assertTrue((scores['market'] == 'US').all())

        top_pick = pick_top_stocks(scores, top_n=1)
        self.assertEqual(len(top_pick), 1)

        one_stock = pick_one_stock(scores)
        self.assertEqual(one_stock['market'], 'US')

        focus_snapshot = build_focus_snapshot(scores, top_pick.index[0])
        self.assertIsNotNone(focus_snapshot)
        self.assertEqual(int(focus_snapshot['rank']), 1)

    def test_oos_backtest_returns_sharpe_summary(self):
        price_map = {f'TICK{i}': make_ohlcv(seed=i) for i in range(6)}
        backtest_df, summary, _ = run_oos_backtest(
            price_map,
            bar_interval='1d',
            horizon=5,
            min_factor_count=3000,
            top_factor_count=64,
            top_n=1,
            test_size=0.2,
            transaction_cost_bps=10,
        )

        self.assertFalse(backtest_df.empty)
        self.assertIn('sharpe_ratio', summary)
        self.assertIn('equity_curve', backtest_df.columns)

    def test_score_entrypoint_requires_trained_model_artifact(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'factor_model.pkl'

            with self.assertRaises(SystemExit) as ctx:
                resolve_model_path(model_path)

            self.assertIn('train_factor_model', str(ctx.exception))

            model_path.write_bytes(b'placeholder')
            self.assertEqual(resolve_model_path(model_path), str(model_path))


if __name__ == '__main__':
    unittest.main()

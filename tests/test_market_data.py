import tempfile
import unittest

import pandas as pd

from src.market_data.cache import MarketDataCache
from src.market_data.symbols import (
    canonicalize_symbol,
    detect_market,
    to_yfinance_symbol,
)


class MarketDataTests(unittest.TestCase):
    def test_symbol_normalization_and_market_detection(self):
        self.assertEqual(canonicalize_symbol(' aapl '), 'AAPL')
        self.assertEqual(detect_market('AAPL'), 'us')
        self.assertEqual(to_yfinance_symbol('msft'), 'MSFT')

    def test_market_data_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MarketDataCache(cache_dir=tmpdir)
            key = {
                'provider': 'yfinance',
                'symbol': 'AAPL',
                'period': '24mo',
                'interval': '1d',
                'adjust': 'qfq',
            }
            df = pd.DataFrame({'Close': [1.0, 2.0]})
            cache.save(key, df)
            loaded = cache.load(key, max_age_seconds=60)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded['Close'].tolist(), [1.0, 2.0])


if __name__ == '__main__':
    unittest.main()

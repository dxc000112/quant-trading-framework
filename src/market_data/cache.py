import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd


def default_cache_ttl_seconds(interval: str) -> int:
    normalized = str(interval).lower()
    if normalized.endswith('m'):
        return 15 * 60
    if normalized.endswith('h'):
        return 60 * 60
    return 24 * 60 * 60


class MarketDataCache:
    def __init__(self, cache_dir: Optional[str] = None):
        root = cache_dir or os.getenv('MARKET_DATA_CACHE_DIR') or '.cache/market_data'
        self.cache_dir = Path(root)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: dict) -> Path:
        payload = json.dumps(key, sort_keys=True, ensure_ascii=True)
        digest = hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]
        symbol = str(key.get('symbol', 'unknown')).replace('.', '_')
        provider = str(key.get('provider', 'auto'))
        interval = str(key.get('interval', 'na')).replace('/', '_')
        period = str(key.get('period', 'na')).replace('/', '_')
        filename = f'{provider}_{symbol}_{interval}_{period}_{digest}.pkl'
        return self.cache_dir / filename

    def load(self, key: dict, max_age_seconds: Optional[int] = None):
        path = self._path_for_key(key)
        if not path.exists():
            return None

        if max_age_seconds is not None:
            age_seconds = max(0.0, time.time() - path.stat().st_mtime)
            if age_seconds > max_age_seconds:
                return None

        try:
            return pd.read_pickle(path)
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def save(self, key: dict, df: pd.DataFrame):
        path = self._path_for_key(key)
        df.to_pickle(path)
        return path

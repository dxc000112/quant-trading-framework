import os
from dataclasses import dataclass

import yaml


@dataclass
class SpxAnchorSettings:
    symbol: str = 'SPX'
    session_open: str = '09:30'
    session_close: str = '16:00'
    anchor_start_time: str = '10:00'
    update_frequency_minutes: int = 5
    price_bin_size: float = 5.0
    lower_interval_quantile: float = 0.15
    upper_interval_quantile: float = 0.85
    contract_size: int = 100
    intraday_volume_weight: float = 0.25
    front_expiry_days: int = 1
    max_snapshot_staleness_minutes: int = 10
    lookback_rows_for_ranks: int = 800
    min_train_sessions: int = 120
    retrain_every_n_sessions: int = 5
    holdout_fraction: float = 0.2


def _load_config(config_path: str = 'config.yaml'):
    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def load_spx_anchor_settings(config_path: str = 'config.yaml') -> SpxAnchorSettings:
    config = _load_config(config_path)
    model_config = config.get('spx_anchor_model') or {}

    settings = SpxAnchorSettings()
    for key, value in model_config.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

    return settings

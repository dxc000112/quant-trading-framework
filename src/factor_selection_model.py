import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import fetch_data
from src.factor_library import generate_factor_library

DEFAULT_SEMICONDUCTOR_UNIVERSE = [
    'TSM', 'NVDA', 'AMD', 'AVGO', 'QCOM', 'MU', 'ASML', 'AMAT', 'LRCX', 'KLAC',
    'INTC', 'ARM', 'ADI', 'TXN', 'MCHP', 'NXPI', 'ON', 'MRVL', 'MPWR', 'TER',
    'GFS', 'UMC', 'STM', 'SWKS', 'QRVO',
]

DEFAULT_US_UNIVERSE = DEFAULT_SEMICONDUCTOR_UNIVERSE

UNIVERSE_PRESETS = {
    'semis': DEFAULT_SEMICONDUCTOR_UNIVERSE,
    'us': DEFAULT_US_UNIVERSE,
}

DEFAULT_FACTOR_MODEL_SETTINGS = {
    'universe_preset': 'semis',
    'lookback_period': '24mo',
    'bar_interval': '1d',
    'forward_horizon': 5,
    'min_factor_count': 3000,
    'top_factor_count': 256,
    'top_n': 1,
    'min_rows': 260,
    'test_size': 0.2,
    'transaction_cost_bps': 10,
    'focus_symbol': 'TSM',
}


def _dedupe_tickers(tickers):
    seen = set()
    output = []
    for ticker in tickers or []:
        normalized = str(ticker).strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _load_config(config_path='config.yaml'):
    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def _infer_market(ticker: str) -> str:
    return 'US'


def load_factor_model_settings(config_path='config.yaml'):
    config = _load_config(config_path)
    factor_config = config.get('factor_model') or {}

    settings = DEFAULT_FACTOR_MODEL_SETTINGS.copy()
    for key in DEFAULT_FACTOR_MODEL_SETTINGS:
        if key in factor_config:
            settings[key] = factor_config[key]

    preset_name = str(settings['universe_preset']).lower()
    preset_universe = UNIVERSE_PRESETS.get(preset_name, DEFAULT_US_UNIVERSE)

    explicit_universe = factor_config.get('universe')
    legacy_stocks = config.get('stocks')
    if explicit_universe:
        resolved_universe = explicit_universe
    elif factor_config:
        resolved_universe = preset_universe
    else:
        resolved_universe = legacy_stocks or preset_universe
    settings['universe'] = _dedupe_tickers(resolved_universe)

    return settings


def load_stock_universe(config_path='config.yaml', fallback=None):
    fallback = fallback or DEFAULT_US_UNIVERSE
    settings = load_factor_model_settings(config_path=config_path)
    return settings.get('universe') or _dedupe_tickers(fallback)


def get_focus_symbol(config_path='config.yaml'):
    settings = load_factor_model_settings(config_path=config_path)
    focus_symbol = settings.get('focus_symbol') or 'TSM'
    return str(focus_symbol).strip().upper()


def fetch_universe_history(tickers, period='24mo', interval='1d', source='auto', min_rows=260):
    price_map = {}

    for ticker in tickers:
        df = fetch_data(ticker, period=period, interval=interval, source=source)
        if df is None or len(df) < min_rows:
            print(f"Skipping {ticker}: insufficient history ({0 if df is None else len(df)} rows).")
            continue

        normalized = df.copy()
        normalized.index = pd.to_datetime(normalized.index)
        if getattr(normalized.index, 'tz', None) is not None:
            normalized.index = normalized.index.tz_localize(None)
        price_map[ticker] = normalized.sort_index()

    return price_map


def build_factor_panel(price_map, min_factor_count=3000):
    if not price_map:
        raise ValueError("No price history available to build a factor panel.")

    frames = []
    feature_columns = None

    for ticker, df in price_map.items():
        factor_frame = generate_factor_library(df, min_factor_count=min_factor_count)
        factor_frame = factor_frame.replace([np.inf, -np.inf], np.nan)

        local = factor_frame.copy()
        local['close'] = df['Close'].reindex(factor_frame.index)
        local['date'] = factor_frame.index
        local['ticker'] = ticker
        frames.append(local.reset_index(drop=True))

        if feature_columns is None:
            feature_columns = factor_frame.columns.tolist()

    panel = pd.concat(frames, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.drop_duplicates(subset=['date', 'ticker'])
    panel = panel.set_index(['date', 'ticker']).sort_index()

    return panel, feature_columns


def prepare_training_panel(price_map, horizon=5, min_factor_count=3000):
    panel, feature_columns = build_factor_panel(price_map, min_factor_count=min_factor_count)

    panel['forward_return'] = panel.groupby(level=1)['close'].shift(-horizon) / panel['close'] - 1
    panel['target_rank'] = panel.groupby(level=0)['forward_return'].rank(pct=True) - 0.5

    cross_section_size = panel.groupby(level=0).size()
    minimum_cross_section = max(2, min(3, len(price_map)))
    valid_dates = cross_section_size[cross_section_size >= minimum_cross_section].index
    panel = panel[panel.index.get_level_values(0).isin(valid_dates)]

    xs_features = panel[feature_columns]
    xs_mean = xs_features.groupby(level=0).transform('mean')
    xs_std = xs_features.groupby(level=0).transform('std').replace(0, np.nan)
    panel[feature_columns] = ((xs_features - xs_mean) / xs_std).clip(-5, 5)

    min_ready_features = max(48, min_factor_count // 50)
    panel = panel[panel[feature_columns].notna().sum(axis=1) >= min_ready_features]
    panel = panel.dropna(subset=['target_rank'])

    return panel, feature_columns


def _split_train_test_dates(panel: pd.DataFrame, test_size: float):
    unique_dates = panel.index.get_level_values(0).unique().sort_values()
    if len(unique_dates) < 30:
        raise ValueError("Need at least 30 dates to train a robust cross-sectional model.")

    split_idx = int(len(unique_dates) * (1 - test_size))
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)

    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    return train_dates, test_dates


def _compute_daily_rank_ic(predictions: pd.Series, target: pd.Series) -> pd.Series:
    joined = pd.DataFrame({'pred': predictions, 'target': target}).dropna()
    if joined.empty:
        return pd.Series(dtype=float)

    def calc_ic(frame: pd.DataFrame):
        if len(frame) < 2:
            return np.nan
        return frame['pred'].rank().corr(frame['target'].rank())

    return joined.groupby(level=0).apply(calc_ic).dropna()


def _compute_daily_top_n_return(predictions: pd.Series, forward_returns: pd.Series, top_n: int = 3) -> pd.Series:
    joined = pd.DataFrame({'pred': predictions, 'forward_return': forward_returns}).dropna()
    if joined.empty:
        return pd.Series(dtype=float)

    def calc_top_return(frame: pd.DataFrame):
        picks = frame.nlargest(min(top_n, len(frame)), 'pred')
        return picks['forward_return'].mean()

    return joined.groupby(level=0).apply(calc_top_return).dropna()


def split_train_test_panel(panel: pd.DataFrame, test_size: float = 0.2):
    train_dates, test_dates = _split_train_test_dates(panel, test_size=test_size)
    train_mask = panel.index.get_level_values(0).isin(train_dates)
    test_mask = panel.index.get_level_values(0).isin(test_dates)
    return panel[train_mask], panel[test_mask], train_dates, test_dates


def fit_factor_model(
    train_panel: pd.DataFrame,
    feature_columns,
    min_factor_count=3000,
    horizon=5,
    top_factor_count=256,
    top_n=1,
):
    feature_scores = (
        train_panel[feature_columns]
        .corrwith(train_panel['target_rank'])
        .abs()
        .sort_values(ascending=False)
        .fillna(0.0)
    )
    selected_features = feature_scores.head(min(top_factor_count, len(feature_scores))).index.tolist()

    if not selected_features:
        raise ValueError("No features were selected for training.")

    fill_values = train_panel[selected_features].median().fillna(0.0)
    X_train = train_panel[selected_features].fillna(fill_values)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, train_panel['target_rank'])

    return {
        'model': model,
        'selected_features': selected_features,
        'feature_scores': feature_scores.head(100).to_dict(),
        'fill_values': fill_values.to_dict(),
        'min_factor_count': min_factor_count,
        'horizon': horizon,
        'top_n': top_n,
    }


def predict_factor_scores(model_bundle, panel_slice: pd.DataFrame) -> pd.Series:
    selected_features = model_bundle['selected_features']
    fill_values = pd.Series(model_bundle['fill_values'])
    X = panel_slice[selected_features].fillna(fill_values).fillna(0.0)
    return pd.Series(model_bundle['model'].predict(X), index=panel_slice.index, name='score')


def train_factor_selection_model(
    price_map,
    horizon=5,
    min_factor_count=3000,
    top_factor_count=256,
    top_n=1,
    test_size=0.2,
    save_path='factor_model.pkl',
):
    panel, feature_columns = prepare_training_panel(
        price_map,
        horizon=horizon,
        min_factor_count=min_factor_count,
    )

    train_panel, test_panel, train_dates, test_dates = split_train_test_panel(
        panel,
        test_size=test_size,
    )

    bundle = fit_factor_model(
        train_panel,
        feature_columns,
        min_factor_count=min_factor_count,
        horizon=horizon,
        top_factor_count=top_factor_count,
        top_n=top_n,
    )

    predictions = predict_factor_scores(bundle, test_panel)
    rank_ic = _compute_daily_rank_ic(predictions, test_panel['target_rank'])
    selected_returns = _compute_daily_top_n_return(predictions, test_panel['forward_return'], top_n=top_n)

    bundle['metrics'] = {
        'train_rows': int(len(train_panel)),
        'test_rows': int(len(test_panel)),
        'train_dates': int(len(train_dates)),
        'test_dates': int(len(test_dates)),
        'selected_features': int(len(bundle['selected_features'])),
        'mean_rank_ic': float(rank_ic.mean()) if not rank_ic.empty else None,
        'median_rank_ic': float(rank_ic.median()) if not rank_ic.empty else None,
        'mean_selected_forward_return': float(selected_returns.mean()) if not selected_returns.empty else None,
    }
    bundle['universe'] = sorted(price_map.keys())

    if save_path:
        joblib.dump(bundle, save_path)

    return bundle


def load_factor_model(model_path='factor_model.pkl'):
    return joblib.load(model_path)


def score_latest_cross_section(model_bundle, price_map):
    panel, _ = build_factor_panel(price_map, min_factor_count=model_bundle['min_factor_count'])

    cross_section_size = panel.groupby(level=0).size()
    valid_dates = cross_section_size[cross_section_size >= max(2, min(3, len(price_map)))].index
    if len(valid_dates) == 0:
        raise ValueError("No cross-sectional date has enough stocks to score.")

    as_of_date = valid_dates.max()
    snapshot = panel.xs(as_of_date, level=0).copy()
    selected_features = model_bundle['selected_features']
    xs_features = snapshot[selected_features]
    xs_mean = xs_features.mean(axis=0)
    xs_std = xs_features.std(axis=0).replace(0, np.nan)
    xs_features = ((xs_features - xs_mean) / xs_std).clip(-5, 5)

    X_live = xs_features.fillna(pd.Series(model_bundle['fill_values'])).fillna(0.0)

    scored = snapshot[['close']].copy()
    scored['score'] = model_bundle['model'].predict(X_live)
    scored['as_of_date'] = as_of_date
    scored.index.name = 'ticker'
    scored['market'] = [_infer_market(ticker) for ticker in scored.index]

    return scored.sort_values('score', ascending=False)


def pick_top_stocks(scored: pd.DataFrame, top_n: int = 1) -> pd.DataFrame:
    limit = max(1, int(top_n))
    return scored.head(limit).copy()


def pick_one_stock(scored: pd.DataFrame) -> pd.Series:
    return pick_top_stocks(scored, top_n=1).iloc[0]


def build_focus_snapshot(scored: pd.DataFrame, focus_symbol: str):
    normalized = str(focus_symbol).strip().upper()
    snapshot = scored.copy()
    snapshot.index = snapshot.index.astype(str).str.upper()

    if normalized not in snapshot.index:
        return None

    rank = int(snapshot.index.get_loc(normalized)) + 1
    total = int(len(snapshot))
    row = snapshot.loc[normalized].copy()
    row['ticker'] = normalized
    row['rank'] = rank
    row['universe_size'] = total
    row['rank_pct'] = rank / total if total else np.nan
    row['is_top_pick'] = rank == 1
    return row

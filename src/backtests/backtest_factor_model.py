import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.factor_selection_model import (
    fetch_universe_history,
    fit_factor_model,
    get_focus_symbol,
    load_factor_model_settings,
    load_stock_universe,
    pick_top_stocks,
    predict_factor_scores,
    prepare_training_panel,
    split_train_test_panel,
)


def periods_per_year(bar_interval: str, holding_bars: int) -> float:
    base = {
        '1d': 252,
        '1wk': 52,
        '1mo': 12,
        '1h': 252 * 6.5,
        '15m': 252 * 26,
        '5m': 252 * 78,
        '1m': 252 * 390,
    }.get(bar_interval, 252)
    return max(base / max(holding_bars, 1), 1.0)


def summarize_backtest(period_returns: pd.Series, annualization_periods: float):
    period_returns = period_returns.dropna()
    if period_returns.empty:
        raise ValueError("No period returns available to summarize.")

    avg_return = period_returns.mean()
    vol = period_returns.std()
    risk_free_rate = 0.02
    rf_period = risk_free_rate / annualization_periods
    sharpe = 0.0 if vol == 0 else ((avg_return - rf_period) / vol) * np.sqrt(annualization_periods)

    equity_curve = (1 + period_returns).cumprod()
    peak = equity_curve.cummax()
    drawdown = equity_curve / peak - 1

    total_return = equity_curve.iloc[-1] - 1
    years = len(period_returns) / annualization_periods
    cagr = (equity_curve.iloc[-1] ** (1 / years) - 1) if years > 0 else total_return

    return {
        'periods': int(len(period_returns)),
        'avg_period_return': float(avg_return),
        'period_volatility': float(vol),
        'annualized_return': float(cagr),
        'annualized_volatility': float(vol * np.sqrt(annualization_periods)),
        'sharpe_ratio': float(sharpe),
        'total_return': float(total_return),
        'max_drawdown': float(drawdown.min()),
        'win_rate': float((period_returns > 0).mean()),
    }


def run_oos_backtest(
    price_map,
    bar_interval='1d',
    horizon=5,
    min_factor_count=3000,
    top_factor_count=256,
    top_n=1,
    test_size=0.2,
    transaction_cost_bps=10,
):
    panel, feature_columns = prepare_training_panel(
        price_map,
        horizon=horizon,
        min_factor_count=min_factor_count,
    )
    train_panel, test_panel, train_dates, test_dates = split_train_test_panel(panel, test_size=test_size)

    model_bundle = fit_factor_model(
        train_panel,
        feature_columns,
        min_factor_count=min_factor_count,
        horizon=horizon,
        top_factor_count=top_factor_count,
        top_n=top_n,
    )

    predictions = predict_factor_scores(model_bundle, test_panel)
    scored_test = test_panel[['close', 'forward_return']].copy()
    scored_test['score'] = predictions

    rebalance_dates = pd.Index(test_dates[::max(horizon, 1)])
    records = []
    previous_picks = set()
    cost_rate = transaction_cost_bps / 10000.0

    for rebalance_date in rebalance_dates:
        if rebalance_date not in scored_test.index.get_level_values(0):
            continue

        day_slice = scored_test.xs(rebalance_date, level=0).dropna(subset=['score', 'forward_return'])
        if day_slice.empty:
            continue

        picks = pick_top_stocks(day_slice.sort_values('score', ascending=False), top_n=top_n)
        current_picks = set(picks.index.tolist())
        buy_count = len(current_picks - previous_picks)
        sell_count = len(previous_picks - current_picks)
        turnover_sides = buy_count + sell_count
        transaction_cost = turnover_sides * cost_rate

        gross_return = picks['forward_return'].mean()
        net_return = gross_return - transaction_cost

        records.append({
            'rebalance_date': rebalance_date,
            'pick_count': int(len(picks)),
            'picked_tickers': ','.join(picks.index.tolist()),
            'gross_return': float(gross_return),
            'transaction_cost': float(transaction_cost),
            'net_return': float(net_return),
            'avg_score': float(picks['score'].mean()),
        })
        previous_picks = current_picks

    if not records:
        raise ValueError("Backtest produced no valid rebalance records.")

    backtest_df = pd.DataFrame(records)
    backtest_df['rebalance_date'] = pd.to_datetime(backtest_df['rebalance_date'])
    backtest_df = backtest_df.sort_values('rebalance_date').reset_index(drop=True)
    backtest_df['equity_curve'] = (1 + backtest_df['net_return']).cumprod()

    summary = summarize_backtest(
        backtest_df['net_return'],
        annualization_periods=periods_per_year(bar_interval, horizon),
    )
    summary.update({
        'train_dates': int(len(train_dates)),
        'test_dates': int(len(test_dates)),
        'top_n': int(top_n),
        'holding_bars': int(horizon),
        'transaction_cost_bps': float(transaction_cost_bps),
        'average_turnover_sides': float(
            (backtest_df['transaction_cost'] / cost_rate).mean()
        ) if cost_rate > 0 else 0.0,
    })

    return backtest_df, summary, model_bundle


def main():
    settings = load_factor_model_settings()
    tickers = load_stock_universe()
    focus_symbol = get_focus_symbol()

    print("--- OOS Backtest: 3000-Factor Stock Selection ---")
    print(
        f"Preset: {settings['universe_preset']} | Universe size: {len(tickers)} | "
        f"Interval: {settings['bar_interval']} | Horizon: {settings['forward_horizon']} | "
        f"Top N: {settings['top_n']} | Cost: {settings['transaction_cost_bps']} bps/side | "
        f"Focus: {focus_symbol}"
    )

    price_map = fetch_universe_history(
        tickers,
        period=settings['lookback_period'],
        interval=settings['bar_interval'],
        source='auto',
        min_rows=settings['min_rows'],
    )
    if len(price_map) < 2:
        raise ValueError("Need at least two stocks with usable history to run the backtest.")

    backtest_df, summary, _ = run_oos_backtest(
        price_map,
        bar_interval=settings['bar_interval'],
        horizon=settings['forward_horizon'],
        min_factor_count=settings['min_factor_count'],
        top_factor_count=settings['top_factor_count'],
        top_n=settings['top_n'],
        test_size=settings['test_size'],
        transaction_cost_bps=settings['transaction_cost_bps'],
    )

    print("\nBacktest Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nLatest Backtest Rows:")
    print(backtest_df.tail(10).to_string(index=False))

    focus_hits = int((backtest_df['picked_tickers'] == focus_symbol).sum())
    print(f"\nFocus Symbol Pick Count ({focus_symbol}): {focus_hits}/{len(backtest_df)}")

    output_dir = Path('backtest_outputs')
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / 'factor_model_backtest.csv'
    json_path = output_dir / 'factor_model_backtest_summary.json'
    backtest_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f"\nSaved details to {csv_path}")
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()

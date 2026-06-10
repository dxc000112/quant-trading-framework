"""
一键跑所有子系统回测并输出结果报告。
- 系统1: SPX Close Anchor (读已有 backtest_outputs)
- 系统2: SPY Reversal Bot (修复 timezone 问题后重跑)
- 系统3: 3000因子半导体选股 (重跑因子模型 OOS 回测)
"""
import sys, os, json, datetime, warnings
warnings.filterwarnings("ignore")

# 确保项目根目录在 path 里
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def sep(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# ─────────────────────────────────────────────
# 系统1: SPX Close Anchor — 读已有结果
# ─────────────────────────────────────────────
sep("系统1: SPX Close Anchor Model (已有回测结果)")

anchor_dir = os.path.join(ROOT, "backtest_outputs", "spx_anchor_versions")
for ver in ["v1", "v2", "v3"]:
    p = os.path.join(anchor_dir, f"{ver}_summary.json")
    try:
        d = json.loads(open(p).read())
        print(f"\n  [{ver.upper()}] sessions={d['sessions']}  rows={d['rows']}")
        print(f"    target_MAE         = {d['target_mae']:.2f} SPX pts")
        print(f"    target_RMSE        = {d['target_rmse']:.2f} SPX pts")
        print(f"    hit_rate ±5pts     = {d['hit_rate_5pts']*100:.1f}%")
        print(f"    hit_rate ±10pts    = {d['hit_rate_10pts']*100:.1f}%")
        print(f"    interval_coverage  = {d['interval_coverage']*100:.1f}%")
        print(f"    avg_interval_width = {d['avg_interval_width']:.1f} pts")
        print(f"    avg_confidence     = {d['avg_confidence_score']:.3f}")
        print(f"    baseline(spot)_MAE = {d['spot_baseline_mae']:.2f}  [naive: 用当前价格预测收盘]")
        print(f"    baseline(gexpain)  = {d['gex_pain_baseline_mae']:.2f}  [GEX Pain 作为预测]")
    except Exception as e:
        print(f"  [{ver}] 读取失败: {e}")

# ─────────────────────────────────────────────
# 系统2: SPY Reversal Bot — 修复 timezone 重跑
# ─────────────────────────────────────────────
sep("系统2: SPY Reversal Bot (ML 过滤反转策略)")

try:
    import numpy as np
    import pandas as pd
    import joblib
    from zoneinfo import ZoneInfo  # Python 3.9+ 标准库，不依赖 pytz

    # ── 加载模型 ──
    model_path = os.path.join(ROOT, "rf_model_short.pkl")
    model_data = joblib.load(model_path)
    model = model_data['model'] if isinstance(model_data, dict) else model_data
    print(f"  模型加载成功: {type(model).__name__}")

    # ── 拉数据 (最近 20 个交易日, 5m 频率) ──
    from src.data_loader import fetch_data
    from src.labeling import get_daily_vol
    from src.meta_model import get_features

    print("  正在下载 SPY 5m 数据 (20d)...")
    df_raw = fetch_data("SPY", period="20d", interval="5m")
    if df_raw is None or df_raw.empty:
        raise ValueError("SPY 数据拉取失败")

    # ── 修复 timezone (不用 pytz, 用 zoneinfo) ──
    if df_raw.index.tz is None:
        df_raw.index = df_raw.index.tz_localize("UTC")
    ny_tz = ZoneInfo("America/New_York")
    df_raw.index = df_raw.index.map(lambda ts: ts.astimezone(ny_tz))
    df_raw.index = pd.DatetimeIndex([
        pd.Timestamp(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second,
                     tzinfo=ts.tzinfo) for ts in df_raw.index
    ])
    print(f"  数据范围: {df_raw.index[0].date()} → {df_raw.index[-1].date()}  ({len(df_raw)} bars)")

    # ── 计算特征 & 置信度 ──
    vol = get_daily_vol(df_raw['Close'])
    X = get_features(df_raw, vol).fillna(0)
    df_raw = df_raw.loc[X.index].copy()
    probs = model.predict_proba(X)
    df_raw['conf'] = probs[:, 1]  # short confidence

    # ── 回测参数 (和 live_bot 一致) ──
    TP_PCT = 0.008
    SL_PCT = 0.005
    ENTRY_CONF_THRESHOLD = 0.20  # conf < 0.20 才做多(反转)
    OPEN_START = datetime.time(9, 30)
    OPEN_END   = datetime.time(11, 0)

    # ── 按交易日回测 ──
    all_trades = []
    dates = sorted(set(df_raw.index.date))

    for d in dates:
        day_df = df_raw[df_raw.index.date == d].copy()
        # 只保留开盘 ~ 收盘时间
        day_df = day_df.between_time("09:30", "15:55")
        if len(day_df) < 20:
            continue

        in_pos = False
        entry_px = tp_px = sl_px = 0.0

        for row in day_df.itertuples():
            price = row.Close
            conf  = getattr(row, 'conf', 0.5)

            if in_pos:
                # SL / TP / EOD
                if row.Low <= sl_px:
                    pnl = sl_px - entry_px
                    all_trades.append({'date': d, 'result': 'SL', 'pnl': pnl})
                    in_pos = False
                elif row.High >= tp_px:
                    pnl = tp_px - entry_px
                    all_trades.append({'date': d, 'result': 'TP', 'pnl': pnl})
                    in_pos = False
                elif row.Index.time() >= datetime.time(15, 55):
                    pnl = price - entry_px
                    all_trades.append({'date': d, 'result': 'EOD', 'pnl': pnl})
                    in_pos = False
                continue

            t = row.Index.time()
            if OPEN_START <= t < OPEN_END and conf < ENTRY_CONF_THRESHOLD and not in_pos:
                entry_px = price
                tp_px    = entry_px * (1 + TP_PCT)
                sl_px    = entry_px * (1 - SL_PCT)
                in_pos   = True

    if not all_trades:
        print("  ⚠️  回测期间没有触发任何交易 (阈值 conf < 0.20 未满足)")
        print(f"  conf 分布: min={df_raw['conf'].min():.3f}  "
              f"median={df_raw['conf'].median():.3f}  "
              f"max={df_raw['conf'].max():.3f}")
    else:
        tdf = pd.DataFrame(all_trades)
        n_trades   = len(tdf)
        n_win      = (tdf['pnl'] > 0).sum()
        win_rate   = n_win / n_trades
        total_pnl  = tdf['pnl'].sum()
        avg_pnl    = tdf['pnl'].mean()
        avg_win    = tdf[tdf['pnl']>0]['pnl'].mean() if n_win else 0
        avg_loss   = tdf[tdf['pnl']<=0]['pnl'].mean() if (n_trades-n_win) else 0
        result_cts = tdf['result'].value_counts().to_dict()
        # 近似年化 (以 20 交易日样本推算, 非严格)
        n_days     = len(dates)
        trades_per_day = n_trades / n_days if n_days else 0

        print(f"\n  回测区间: {dates[0]} → {dates[-1]}  ({n_days} 交易日)")
        print(f"  总交易次数 : {n_trades}  (平均 {trades_per_day:.1f} 次/天)")
        print(f"  胜率       : {win_rate*100:.1f}%")
        print(f"  总 PnL     : ${total_pnl:.2f} / share")
        print(f"  平均 PnL   : ${avg_pnl:.3f} / trade")
        print(f"  平均盈利   : ${avg_win:.3f}  | 平均亏损: ${avg_loss:.3f}")
        print(f"  盈亏比     : {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "  盈亏比: N/A")
        print(f"  结果分布   : {result_cts}")

        # 置信度分布
        print(f"\n  [模型置信度] conf < 0.20 bars 占比: "
              f"{(df_raw['conf'] < 0.20).mean()*100:.1f}%  "
              f"| median conf: {df_raw['conf'].median():.3f}")

except Exception as e:
    import traceback
    print(f"  ❌ 系统2 失败: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────
# 系统3: 3000因子选股 — 重跑 OOS 回测
# ─────────────────────────────────────────────
sep("系统3: 3000因子半导体选股 (重跑 OOS 回测)")

try:
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
    from src.backtests.backtest_factor_model import run_oos_backtest

    settings    = load_factor_model_settings()
    tickers     = load_stock_universe()
    focus       = get_focus_symbol()

    print(f"  Universe: {settings['universe_preset']} ({len(tickers)} 只股票)")
    print(f"  Interval={settings['bar_interval']}  Horizon={settings['forward_horizon']}d  "
          f"TopN={settings['top_n']}  Cost={settings['transaction_cost_bps']}bps  Focus={focus}")
    print("  正在下载 24 个月历史数据 (需要 1-3 分钟)...")

    price_map = fetch_universe_history(
        tickers,
        period=settings['lookback_period'],
        interval=settings['bar_interval'],
        source='auto',
        min_rows=settings['min_rows'],
    )
    print(f"  成功加载 {len(price_map)} 只股票的历史数据")

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

    rebal_dates = pd.to_datetime(backtest_df['rebalance_date'])
    print(f"\n  回测区间 : {rebal_dates.min().date()} → {rebal_dates.max().date()}")
    print(f"  再平衡次数: {summary['periods']}  (每 {settings['forward_horizon']} 日换仓)")
    print(f"  训练期    : {summary['train_dates']} bars  测试期: {summary['test_dates']} bars")
    print(f"\n  ── 核心指标 ──")
    print(f"  年化收益率 : {summary['annualized_return']*100:.1f}%")
    print(f"  年化波动率 : {summary['annualized_volatility']*100:.1f}%")
    print(f"  Sharpe     : {summary['sharpe_ratio']:.3f}")
    print(f"  最大回撤   : {summary['max_drawdown']*100:.1f}%")
    print(f"  总收益     : {summary['total_return']*100:.1f}%")
    print(f"  胜率       : {summary['win_rate']*100:.1f}%")
    print(f"  平均换手   : {summary['average_turnover_sides']:.2f} sides/period")

    print(f"\n  最近10次换仓记录:")
    cols = ['rebalance_date', 'picked_tickers', 'gross_return', 'net_return', 'equity_curve']
    print(backtest_df[cols].tail(10).to_string(index=False))

    # 保存新结果
    out_dir = os.path.join(ROOT, "backtest_outputs")
    backtest_df.to_csv(os.path.join(out_dir, "factor_model_backtest_fresh.csv"), index=False)
    with open(os.path.join(out_dir, "factor_model_backtest_fresh_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  结果已保存到 backtest_outputs/factor_model_backtest_fresh*.csv/json")

except Exception as e:
    import traceback
    print(f"  ❌ 系统3 失败: {e}")
    traceback.print_exc()

sep("全部完成")

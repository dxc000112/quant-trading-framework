"""
Backtest: Sparse (15m) vs Original (5m) Strategy
Replays both strategies over the past week to compare performance.
Run from project root:
    python -m src.backtests.replay_sparse_vs_original
"""
import pandas as pd
import numpy as np
import joblib
import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_loader import fetch_data
from src.labeling import get_daily_vol
from src.meta_model import get_features


def calc_vwap_and_sma(df):
    """Calculate daily VWAP and SMA-50, same as live bot logic."""
    df = df.copy()
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['vol_x_price'] = df['Volume'] * df['typical_price']

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df['local_time'] = df.index.tz_convert('America/New_York')
    df['date'] = df['local_time'].dt.date

    daily_vwap = []
    for date in df['date'].unique():
        df_day = df[df['date'] == date].copy()
        df_day['cum_vol_price'] = df_day['vol_x_price'].cumsum()
        df_day['cum_vol'] = df_day['Volume'].cumsum()
        df_day['vwap_daily'] = df_day['cum_vol_price'] / df_day['cum_vol']
        daily_vwap.append(df_day['vwap_daily'])

    df['vwap'] = pd.concat(daily_vwap)
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    return df


def replay_strategy(df, model, strategy_name, interval,
                    long_conf=0.20, short_conf=0.55, exit_conf=0.41,
                    tp_pct=0.008, sl_pct=0.005,
                    apply_filters=True):
    """
    Replays a strategy day-by-day.
    Returns a summary dict + trade log.
    """
    # Calculate features
    try:
        vol = get_daily_vol(df['Close'])
        X = get_features(df, vol)
        df = df.loc[X.index].copy()
        X = X.fillna(0)
        probs = model.predict_proba(X)
        df['conf'] = probs[:, 1]
    except Exception as e:
        print(f"  ❌ Feature Error ({strategy_name}): {e}")
        return None, []

    # Calculate VWAP & SMA50 for filters
    df = calc_vwap_and_sma(df)

    # Convert to NY time
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df_ny = df.copy()
    df_ny.index = df_ny.index.tz_convert('America/New_York')

    # Get trading days
    trading_days = sorted(df_ny.index.date)
    unique_days = sorted(set(trading_days))

    all_trades = []
    daily_results = []

    for day in unique_days:
        day_df = df_ny[df_ny.index.date == day]
        if day_df.empty:
            continue

        day_name = pd.Timestamp(day).strftime('%A')

        in_position = False
        position_type = None  # 'long' or 'short'
        entry_price = 0.0
        tp_price = 0.0
        sl_price = 0.0
        prev_conf = None
        day_pnl = 0.0
        day_trades = 0
        day_wins = 0

        for row in day_df.itertuples():
            price = row.Close
            conf = row.conf
            t = row.Index.time()
            curr_vwap = row.vwap if hasattr(row, 'vwap') and not pd.isna(row.vwap) else price
            curr_sma50 = row.sma_50 if hasattr(row, 'sma_50') and not pd.isna(row.sma_50) else price

            is_open = (t >= datetime.time(9, 30) and t < datetime.time(11, 0))
            is_valid_day = (day_name != 'Friday') if apply_filters else True
            is_uptrend = (price > curr_vwap) and (price > curr_sma50) if apply_filters else True

            # --- EXIT LOGIC ---
            if in_position:
                if position_type == 'long':
                    # TP
                    if row.High >= tp_price:
                        pnl = tp_price - entry_price
                        day_pnl += pnl
                        day_wins += 1
                        all_trades.append({
                            'date': day, 'time': t.strftime('%H:%M'),
                            'type': 'LONG', 'exit': 'TP',
                            'entry': entry_price, 'exit_price': tp_price,
                            'pnl': pnl, 'conf_entry': conf
                        })
                        in_position = False
                    # SL
                    elif row.Low <= sl_price:
                        pnl = sl_price - entry_price
                        day_pnl += pnl
                        all_trades.append({
                            'date': day, 'time': t.strftime('%H:%M'),
                            'type': 'LONG', 'exit': 'SL',
                            'entry': entry_price, 'exit_price': sl_price,
                            'pnl': pnl, 'conf_entry': conf
                        })
                        in_position = False
                    # Dynamic exit
                    elif conf >= exit_conf:
                        pnl = price - entry_price
                        day_pnl += pnl
                        if pnl > 0:
                            day_wins += 1
                        all_trades.append({
                            'date': day, 'time': t.strftime('%H:%M'),
                            'type': 'LONG', 'exit': 'DYN',
                            'entry': entry_price, 'exit_price': price,
                            'pnl': pnl, 'conf_entry': conf
                        })
                        in_position = False
                    # EOD
                    elif t >= datetime.time(15, 55):
                        pnl = price - entry_price
                        day_pnl += pnl
                        if pnl > 0:
                            day_wins += 1
                        all_trades.append({
                            'date': day, 'time': t.strftime('%H:%M'),
                            'type': 'LONG', 'exit': 'EOD',
                            'entry': entry_price, 'exit_price': price,
                            'pnl': pnl, 'conf_entry': conf
                        })
                        in_position = False

                elif position_type == 'short':
                    # Short: TP when price drops
                    if row.Low <= sl_price:  # sl_price here is TP for short (price drop)
                        pnl = entry_price - sl_price
                        day_pnl += pnl
                        day_wins += 1
                        all_trades.append({
                            'date': day, 'time': t.strftime('%H:%M'),
                            'type': 'SHORT', 'exit': 'TP',
                            'entry': entry_price, 'exit_price': sl_price,
                            'pnl': pnl, 'conf_entry': conf
                        })
                        in_position = False
                    elif row.High >= tp_price:  # tp_price here is SL for short (price rise)
                        pnl = entry_price - tp_price
                        day_pnl += pnl
                        all_trades.append({
                            'date': day, 'time': t.strftime('%H:%M'),
                            'type': 'SHORT', 'exit': 'SL',
                            'entry': entry_price, 'exit_price': tp_price,
                            'pnl': pnl, 'conf_entry': conf
                        })
                        in_position = False
                    elif t >= datetime.time(15, 55):
                        pnl = entry_price - price
                        day_pnl += pnl
                        if pnl > 0:
                            day_wins += 1
                        all_trades.append({
                            'date': day, 'time': t.strftime('%H:%M'),
                            'type': 'SHORT', 'exit': 'EOD',
                            'entry': entry_price, 'exit_price': price,
                            'pnl': pnl, 'conf_entry': conf
                        })
                        in_position = False

                prev_conf = conf
                if in_position:
                    continue
                else:
                    day_trades += 1
                    continue

            # --- ENTRY LOGIC ---
            # LONG: Conf < long_conf during morning window
            if is_open and conf < long_conf and is_valid_day and is_uptrend and not in_position:
                entry_price = price
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
                in_position = True
                position_type = 'long'
                day_trades += 1

            # SHORT: Conf >= short_conf (any time, like the sparse bot)
            elif (prev_conf is None or prev_conf < short_conf) and conf >= short_conf and not in_position:
                entry_price = price
                # For short: TP is price dropping, SL is price rising
                tp_price = entry_price * (1 + sl_pct)  # SL for short = price goes up
                sl_price = entry_price * (1 - tp_pct)  # TP for short = price goes down
                in_position = True
                position_type = 'short'
                day_trades += 1

            prev_conf = conf

        # EOD force close
        if in_position and len(day_df) > 0:
            last_price = day_df['Close'].iloc[-1]
            if position_type == 'long':
                pnl = last_price - entry_price
            else:
                pnl = entry_price - last_price
            day_pnl += pnl
            if pnl > 0:
                day_wins += 1
            day_trades += 1
            all_trades.append({
                'date': day, 'time': 'EOD',
                'type': position_type.upper(), 'exit': 'FORCE',
                'entry': entry_price, 'exit_price': last_price,
                'pnl': pnl, 'conf_entry': 0
            })

        daily_results.append({
            'date': day,
            'day': day_name,
            'trades': day_trades,
            'wins': day_wins,
            'pnl': day_pnl,
            'open': day_df['Open'].iloc[0],
            'close': day_df['Close'].iloc[-1],
        })

    return daily_results, all_trades


def print_results(strategy_name, daily_results, all_trades):
    """Pretty print backtest results."""
    print(f"\n{'='*60}")
    print(f"  📊 {strategy_name} — 上周回测结果")
    print(f"{'='*60}")

    if not daily_results:
        print("  ❌ 没有数据")
        return

    total_trades = sum(d['trades'] for d in daily_results)
    total_wins = sum(d['wins'] for d in daily_results)
    total_pnl = sum(d['pnl'] for d in daily_results)

    # Daily breakdown
    print(f"\n  {'日期':<14} {'星期':<10} {'交易数':<8} {'胜':<6} {'当日PnL':>10}")
    print(f"  {'-'*52}")
    for d in daily_results:
        day_emoji = '🟢' if d['pnl'] > 0 else ('🔴' if d['pnl'] < 0 else '⚪')
        mkt_ret = (d['close'] - d['open']) / d['open'] * 100
        print(f"  {str(d['date']):<14} {d['day']:<10} {d['trades']:<8} {d['wins']:<6} {day_emoji} ${d['pnl']:>+8.2f}  (SPY {mkt_ret:>+.2f}%)")

    print(f"  {'-'*52}")
    win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    print(f"  {'总计':<14} {'':10} {total_trades:<8} {total_wins:<6}    ${total_pnl:>+8.2f}")
    print(f"  胜率: {win_rate:.1f}%")

    # Trade log
    if all_trades:
        print(f"\n  --- 交易明细 ---")
        for t in all_trades:
            emoji = '🟢' if t['pnl'] > 0 else '🔴'
            print(f"  {emoji} {t['date']} {t['time']} | {t['type']:<5} | "
                  f"入 ${t['entry']:.2f} → 出 ${t['exit_price']:.2f} | "
                  f"{t['exit']:<4} | PnL: ${t['pnl']:+.2f}")


def main():
    print("=" * 60)
    print("  🔬 回测对比: Sparse (15m) vs Original (5m)")
    print("  上周 (过去 7 个交易日)")
    print("=" * 60)

    # Load Model
    try:
        model_data = joblib.load('rf_model_short.pkl')
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        print("✅ 模型加载成功: rf_model_short.pkl")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # --- Fetch data for BOTH strategies ---
    # Original: 5m bars, ~20d for context
    # Sparse:  15m bars, ~20d for context
    print("\n📡 获取数据中...")

    print("  [1/2] 5分钟数据 (Original Strategy)...")
    df_5m = fetch_data("SPY", period="20d", interval="5m")
    if df_5m is not None:
        print(f"        获取到 {len(df_5m)} 根 5m K线")
    else:
        print("        ❌ 获取5m数据失败")

    print("  [2/2] 15分钟数据 (Sparse Strategy)...")
    df_15m = fetch_data("SPY", period="20d", interval="15m")
    if df_15m is not None:
        print(f"        获取到 {len(df_15m)} 根 15m K线")
    else:
        print("        ❌ 获取15m数据失败")

    # Determine last week's date range
    # Current date: 2026-03-18 (Wednesday)
    # Last week: March 9 (Mon) to March 13 (Fri)
    # But let's dynamically get the last 5 trading days from the data
    if df_5m is not None:
        if df_5m.index.tz is None:
            df_5m.index = df_5m.index.tz_localize('UTC')
        df_5m_ny = df_5m.copy()
        df_5m_ny.index = df_5m_ny.index.tz_convert('America/New_York')
        all_days_5m = sorted(set(df_5m_ny.index.date))

        # Last week = the 5 trading days BEFORE this week
        # This week started March 16 (Mon), so last week = March 9-13
        today = datetime.date(2026, 3, 18)
        last_monday = today - datetime.timedelta(days=today.weekday() + 7)  # March 9
        last_friday = last_monday + datetime.timedelta(days=4)  # March 13

        last_week_days = [d for d in all_days_5m if last_monday <= d <= last_friday]
        print(f"\n📅 上周交易日: {[str(d) for d in last_week_days]}")

        if not last_week_days:
            # Fallback: use last 5 trading days
            last_week_days = all_days_5m[-5:]
            print(f"   (回退到最近5个交易日: {[str(d) for d in last_week_days]})")

    # ===========================
    # STRATEGY 1: ORIGINAL (5m)
    # ===========================
    if df_5m is not None:
        # We need enough context for features, so keep full df for feature calc
        # but only evaluate trades on last week days
        print("\n🔄 回测 Original Strategy (5m)...")
        orig_results, orig_trades = replay_strategy(
            df_5m, model,
            strategy_name="Original (5m)",
            interval="5m",
            long_conf=0.20,
            short_conf=99.0,  # Original doesn't have short signals
            exit_conf=0.41,
            tp_pct=0.008,
            sl_pct=0.005,
            apply_filters=True
        )
        # Filter to last week only
        if orig_results:
            orig_results = [d for d in orig_results if d['date'] in last_week_days]
            orig_trades = [t for t in orig_trades if t['date'] in last_week_days]
        print_results("Original Strategy (5m, Conf<0.20 Long Only)", orig_results, orig_trades)
    else:
        print("\n❌ 跳过 Original 回测 (无5m数据)")

    # ===========================
    # STRATEGY 2: SPARSE (15m)
    # ===========================
    if df_15m is not None:
        print("\n🔄 回测 Sparse Strategy (15m)...")
        sparse_results, sparse_trades = replay_strategy(
            df_15m, model,
            strategy_name="Sparse (15m)",
            interval="15m",
            long_conf=0.20,
            short_conf=0.55,  # Sparse has short signals at 0.55
            exit_conf=0.41,
            tp_pct=0.008,
            sl_pct=0.005,
            apply_filters=True
        )
        # Filter to last week only
        if sparse_results:
            if df_15m.index.tz is None:
                df_15m_check = df_15m.copy()
                df_15m_check.index = df_15m_check.index.tz_localize('UTC')
            else:
                df_15m_check = df_15m
            df_15m_ny = df_15m_check.copy()
            df_15m_ny.index = df_15m_ny.index.tz_convert('America/New_York')
            all_days_15m = sorted(set(df_15m_ny.index.date))
            last_week_days_15m = [d for d in all_days_15m if last_monday <= d <= last_friday]
            if not last_week_days_15m:
                last_week_days_15m = all_days_15m[-5:]

            sparse_results = [d for d in sparse_results if d['date'] in last_week_days_15m]
            sparse_trades = [t for t in sparse_trades if t['date'] in last_week_days_15m]

        print_results("Sparse Strategy (15m, Conf>=0.55 Short + Long)", sparse_results, sparse_trades)
    else:
        print("\n❌ 跳过 Sparse 回测 (无15m数据)")

    # ===========================
    # COMPARISON
    # ===========================
    print(f"\n{'='*60}")
    print(f"  📈 对比总结")
    print(f"{'='*60}")

    if df_5m is not None and orig_results:
        orig_pnl = sum(d['pnl'] for d in orig_results)
        orig_total = sum(d['trades'] for d in orig_results)
        orig_wins = sum(d['wins'] for d in orig_results)
        orig_wr = orig_wins / orig_total * 100 if orig_total > 0 else 0
    else:
        orig_pnl = orig_total = orig_wins = orig_wr = 0

    if df_15m is not None and sparse_results:
        sparse_pnl = sum(d['pnl'] for d in sparse_results)
        sparse_total = sum(d['trades'] for d in sparse_results)
        sparse_wins = sum(d['wins'] for d in sparse_results)
        sparse_wr = sparse_wins / sparse_total * 100 if sparse_total > 0 else 0
    else:
        sparse_pnl = sparse_total = sparse_wins = sparse_wr = 0

    print(f"\n  {'策略':<30} {'交易数':>8} {'胜率':>8} {'总PnL':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Original (5m, Long Only)':<30} {orig_total:>8} {orig_wr:>7.1f}% ${orig_pnl:>+9.2f}")
    print(f"  {'Sparse (15m, Long+Short)':<30} {sparse_total:>8} {sparse_wr:>7.1f}% ${sparse_pnl:>+9.2f}")
    print(f"  {'-'*58}")

    diff = sparse_pnl - orig_pnl
    if diff > 0:
        print(f"  🏆 Sparse 策略领先: ${diff:+.2f}")
    elif diff < 0:
        print(f"  🏆 Original 策略领先: ${-diff:+.2f}")
    else:
        print(f"  🤝 两策略持平")

    print()


if __name__ == "__main__":
    main()

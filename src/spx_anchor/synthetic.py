import numpy as np
import pandas as pd


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
        trend_bias = rng.normal(0, 0.5)
        target_price = base_price + latent * 12 + day_idx * 1.5
        call_wall = round((target_price + 20 + trend_bias * 4) / 5) * 5
        put_wall = round((target_price - 20 + trend_bias * 2) / 5) * 5
        long_gamma = round((target_price + trend_bias * 2) / 5) * 5

        prices = []
        current = target_price - latent * 6
        for idx, _ in enumerate(minute_index):
            if idx < 30:
                drift = 0.18 * latent
            else:
                drift = 0.10 * (target_price - current) + 0.03 * trend_bias
            noise = rng.normal(0, 1.1 + 0.15 * abs(trend_bias))
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
        strikes = np.arange(long_gamma - 50, long_gamma + 55, 5)

        for snapshot_time in snapshot_times:
            minutes_to_close = max((session_date + pd.Timedelta(hours=16) - snapshot_time).total_seconds() / 60.0, 1.0)
            intraday_flow = 1.0 + 0.6 * np.exp(-minutes_to_close / 90.0)
            for strike in strikes:
                call_oi = 80 + 900 * np.exp(-((strike - call_wall) / 7.5) ** 2) + 550 * np.exp(-((strike - long_gamma) / 10.0) ** 2)
                put_oi = 80 + 900 * np.exp(-((strike - put_wall) / 7.5) ** 2)
                call_gamma = 0.010 + 0.020 * np.exp(-((strike - long_gamma) / 12.0) ** 2)
                put_gamma = 0.009 + 0.018 * np.exp(-((strike - put_wall) / 11.0) ** 2)
                iv = 0.14 + 0.01 * abs(latent)
                vanna = 0.0012 * np.exp(-((strike - target_price) / 18.0) ** 2)
                charm = 0.0009 * np.exp(-((strike - target_price) / 20.0) ** 2)

                all_options.append({
                    'as_of': snapshot_time,
                    'expiry': session_date,
                    'option_type': 'C',
                    'strike': float(strike),
                    'gamma': float(call_gamma),
                    'open_interest': float(call_oi),
                    'volume': float(max(call_oi * 0.08 * intraday_flow + rng.normal(0, 8), 1)),
                    'iv': float(iv),
                    'vanna': float(vanna),
                    'charm': float(charm),
                })
                all_options.append({
                    'as_of': snapshot_time,
                    'expiry': session_date,
                    'option_type': 'P',
                    'strike': float(strike),
                    'gamma': float(put_gamma),
                    'open_interest': float(put_oi),
                    'volume': float(max(put_oi * 0.08 * intraday_flow + rng.normal(0, 8), 1)),
                    'iv': float(iv),
                    'vanna': float(vanna * 0.9),
                    'charm': float(charm * 0.9),
                })

            next_day_expiry = session_date + pd.Timedelta(days=1)
            monthly_expiry = session_date + pd.offsets.Week(weekday=4)
            for strike in strikes[::2]:
                for expiry in [next_day_expiry, monthly_expiry]:
                    all_options.append({
                        'as_of': snapshot_time,
                        'expiry': expiry,
                        'option_type': 'C',
                        'strike': float(strike),
                        'gamma': float(0.006 + 0.009 * np.exp(-((strike - long_gamma) / 14.0) ** 2)),
                        'open_interest': float(110 + 220 * np.exp(-((strike - call_wall) / 15.0) ** 2)),
                        'volume': float(max(20 + rng.normal(0, 4), 1)),
                        'iv': float(iv * 1.05),
                        'vanna': float(vanna * 0.7),
                        'charm': float(charm * 0.7),
                    })
                    all_options.append({
                        'as_of': snapshot_time,
                        'expiry': expiry,
                        'option_type': 'P',
                        'strike': float(strike),
                        'gamma': float(0.005 + 0.008 * np.exp(-((strike - put_wall) / 14.0) ** 2)),
                        'open_interest': float(110 + 220 * np.exp(-((strike - put_wall) / 15.0) ** 2)),
                        'volume': float(max(20 + rng.normal(0, 4), 1)),
                        'iv': float(iv * 1.05),
                        'vanna': float(vanna * 0.65),
                        'charm': float(charm * 0.65),
                    })

    return pd.concat(all_bars).sort_index(), pd.DataFrame(all_options)

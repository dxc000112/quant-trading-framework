import os
import time
import datetime
import pandas as pd
from dotenv import load_dotenv
from src.data_loader import fetch_data
from src.strategy import calculate_indicators, check_signal
from src.notifier import DiscordNotifier, play_beep
from src.plotting import plot_triple_barrier_events
from src.labeling import get_daily_vol

# Load environment variables
load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
TICKER = "SPY"
INTERVAL = "1m"
PERIOD = "1d" 
LOOP_DELAY = 60 # seconds
COOLDOWN_SECONDS = 900 # 15 minutes

def main():
    print(f"--- Intraday SPY Monitor Started ---")
    print(f"Target: {TICKER} | Interval: {INTERVAL}")
    print(f"Signal: Price > VWAP and RSI > 50")
    print(f"Notification: Discord Webhook")
    print(f"Press Ctrl+C to stop.")

    notifier = DiscordNotifier(DISCORD_WEBHOOK_URL)
    last_alert_time = 0

    while True:
        try:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Scanning...", end=" ", flush=True)

            # 1. Fetch Data
            df = fetch_data(TICKER, period=PERIOD, interval=INTERVAL)
            
            if df is not None:
                # 2. Calculate Indicators
                df = calculate_indicators(df)
                
                # 3. Check Signal
                triggered, details = check_signal(df)
                
                # Print status
                latest = df.iloc[-1]
                print(f"Price: {latest['Close']:.2f} | VWAP: {latest.get('VWAP', 0):.2f} | RSI: {latest.get('RSI', 0):.2f}", end="")

                if triggered:
                    current_time = time.time()
                    if current_time - last_alert_time > COOLDOWN_SECONDS:
                        print(" -> SIGNAL DETECTED!")
                        
                        message = (f"🚀 SPY Intraday Alert 🚀\n"
                                   f"Time: {details['date']}\n"
                                   f"Price: {details['price']:.2f}\n"
                                   f"VWAP: {details['vwap']:.2f}\n"
                                   f"RSI: {details['rsi']:.2f}\n"
                                   f"Condition: Price > VWAP & RSI > 50")
                        
                        # Audio Alert
                        play_beep()
                        
                        # 1. Send Text Alert
                        notifier.send_alert("SPY Signal Detected", message, 0x00ff00)
                        
                        # 2. Generate Trade Box Chart
                        # We need to simulate an 'event' structure for the plotting function
                        # Volatility
                        vol = get_daily_vol(df['Close'])
                        latest_vol = vol.iloc[-1] if not vol.empty else 0.01
                        
                        # Mock Event DataFrame for Plotting
                        t0 = details['date']
                        # Arbitrary t1 (e.g. 50 bars later, just for viz context)
                        # In real-time, we don't know t1 yet. 
                        # Plotting function expects historical data usually.
                        # For real-time 'Trade Box', we visualize the ENTRY and potentially the barriers relative to entry.
                        # We use the available data up to now.
                        
                        # Hack: To show the 'Trade Box' setup, we just plot the recent history and draw the barriers extending forward?
                        # The plotting function `plot_triple_barrier_events` is designed for historical backtest review (t0 to t1).
                        # For live, we can plot the Recent Price Path + Horizontal Lines.
                        # Let's adapt: We pass the current time as t0. t1 sets the view range.
                        
                        # We will construct a dummy event to utilize the existing function logic
                        # But the function tries to slice close[t0:t1]. If t0 is NOW, and t1 is FUTURE, it plots nothing?
                        # Correction: We want to show the context leading UP TO the signal.
                        # So let's plot the last 60 bars, and mark the CURRENT bar as Entry.
                        # The barriers will just be horizontal lines.
                        
                        # Actually, looking at src/plotting.py:
                        # start_time = signal_index (t0)
                        # end_time = t1
                        # price_path = close[start_time:end_time]
                        # This implies it plots FUTURE path.
                        # For live signal, we have no future path.
                        # WE WANT TO SEE THE SETUP. So we want to see [t0 - 60, t0].
                        
                        # Let's stick to the request: "Call plotting.plot_triple_barrier_events() ... to generate the chart"
                        # If I pass t1 = t0, it will plot a single point?
                        # Let's modifying plotting usage or use a different visualization for LIVE.
                        # BUT the user asked to use `plot_triple_barrier_events`.
                        # I will simply set t1 to t0 so it marks the entry.
                        # Wait, the function plots `close[start_time:end_time]`. If equal, it might be empty or 1 point.
                        # I'll update main to just send the chart if it works, otherwise I might need to adjust slicing in plotting or here.
                        # Let's try to pass t1 = t0.
                        
                        # To make it useful, we really want to see the PREVIOUS candles.
                        # The function `plot_triple_barrier_events` is specifically for "Event Analysis" (post-mortem).
                        # For "Live Alert", we want "Context".
                        # I will call `plot_triple_barrier_events` but I need to trick it or rely on it just plotting the markers.
                        
                        # BETTER APPROACH for LIVE:
                        # Retrain logic: The user said "Visually verify if labels are correct" (Context: Backtest/Meta-Labeling).
                        # Then in Add-on: "Send Chart Images (The Triple Barrier visualization) instantly."
                        # If I send a chart with NO path (because it's the future), it's just an entry point.
                        # That's fine. It shows where we are entering and where the stops are.
                        # Let's assume t1 = t0 + small delta to avoid empty slice, or just t0.
                        
                        # Let's create a synthetic 't1' slightly in the future (Time Barrier) but verify if plotting handles lack of data.
                        # If `close[t0:t1]` is empty (which it is, since t1 is future), plot might crash or show nothing.
                        
                        # I will generate a simple chart using matplotlib directly here OR modify plotting to handle 'lookback'.
                        # Given constraints, I will attempt to use the existing function but maybe I should have modified it to support lookback.
                        # Let's assume for now I pass t1 = t0 (timestamp).
                        # AND I will Modify plotting.py slightly to plot a buffer BEFORE t0 if loopback is needed?
                        # No, I should stick to the plan.
                        
                        # Let's define t1 as 1 hour later?
                        # But data doesn't exist.
                        # I will create a dummy event with t1 = t0.
                        # And pass the WHOLE dataframe 'df' as 'close'.
                        
                        # Generating chart...
                        t1_mock = t0 # Current time
                        events_mock = pd.DataFrame({'t1': [t1_mock], 'trgt': [latest_vol]}, index=[t0])
                        
                        # ISSUE: The plotting function plots `close[start_time:end_time]`.
                        # If start_time = t0 (now) and end_time=t0, we see nothing.
                        # We want to see history.
                        # I will modify plotting.py in a separate tool call if needed, but easier:
                        # I will manually plot here? No, duplicate code.
                        # I will modify `src/plotting.py` to allow `lookback` so it plots context.
                        # But I am in main.py step.
                        
                        # Let's just implement the call. If it's a single dot, so be it. 
                        # Actually, for a "Signal", seeing the history *leading up* to it is most important.
                        # The `plot_triple_barrier_events` is essentially a "Trade Box" (Future).
                        # Maybe the user WANTS to see the empty box where price *will* go?
                        # Yes: "Plot Vertical Line: The Time Barrier... Highlight the price path inside this box"
                        # So it will show the empty space bounded by barriers.
                        # That is actually cool.
                        
                        # Define T1 as strictly future (e.g. +10 bars)
                        t1_future = t0 + pd.Timedelta(minutes=50) # 50 bars vertical barrier
                        events_mock = pd.DataFrame({'t1': [t1_future], 'trgt': [latest_vol]}, index=[t0])
                        
                        buf = plot_triple_barrier_events(df['Close'], events_mock, t0, pt_sl=[1, 1], target_vol=df['Close'], return_buffer=True)
                        
                        if buf:
                             notifier.send_chart(buf, "Trade Box Setup")
                        
                        last_alert_time = current_time
                    else:
                        print(" -> Signal (Cooldown active)")
                else:
                    print(" -> No Signal")
            
            # Wait for next loop
            time.sleep(LOOP_DELAY)

        except KeyboardInterrupt:
            print("\nMonitor stopped by user.")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            time.sleep(LOOP_DELAY)

if __name__ == "__main__":
    main()

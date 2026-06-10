import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io

def plot_triple_barrier_events(close, events, signal_index, pt_sl, target_vol, save_path=None, side=1, **kwargs):
    """
    Plots the 'Trade Box' for a specific event.
    Args:
        side: 1 for Long, -1 for Short.
    """
    if signal_index not in events.index:
        print(f"Signal index {signal_index} not found in events.")
        return

    # Extract event details
    event = events.loc[signal_index]
    t1 = event['t1']
    trgt = event['trgt']
    
    start_time = signal_index
    
    lookback_window = kwargs.get('lookback', 0)
    
    if lookback_window > 0:
         # Find integer index of start_time
         try:
             idx_loc = close.index.get_loc(start_time)
             start_loc = max(0, idx_loc - lookback_window)
             plot_start_time = close.index[start_loc]
         except:
             plot_start_time = start_time
    else:
        plot_start_time = start_time
        
    end_time = t1
    
    # Get Price Path (Historical)
    # We slice up to the last available data point if end_time is future
    last_avail = close.index[-1]
    plot_end_time = min(end_time, last_avail) if end_time > last_avail else end_time
    
    price_path = close[plot_start_time:plot_end_time]
    entry_price = close[start_time] if start_time in close.index else price_path.iloc[-1]
    
    # Calculate Barriers
    # Long: PT = Entry * (1 + PT*Vol), SL = Entry * (1 - SL*Vol)
    # Short: PT = Entry * (1 - PT*Vol), SL = Entry * (1 + SL*Vol)
    
    if side == 1:
        pt_mult = 1
        sl_mult = -1
    else:
        pt_mult = -1
        sl_mult = 1

    if pt_sl[0] > 0:
        pt_price = entry_price * (1 + pt_mult * pt_sl[0] * trgt)
    else:
        pt_price = np.nan
        
    if pt_sl[1] > 0:
        sl_price = entry_price * (1 + sl_mult * pt_sl[1] * trgt)
    else:
        sl_price = np.nan

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(price_path.index, price_path.values, label='Price', color='black', alpha=0.7)
    
    # Entry
    plt.scatter(start_time, entry_price, color='blue', label='Entry', marker='^', s=100)
    
    # Barriers
    plt.axhline(pt_price, color='green', linestyle='--', label='Profit Taking')
    plt.axhline(sl_price, color='red', linestyle='--', label='Stop Loss')
    plt.axvline(end_time, color='gray', linestyle='--', label='Time Barrier')
    
    # Outcome
    exit_price =  close[end_time] if end_time in close.index else price_path.iloc[-1]
    
    # Determine touch type for visualization
    # Side 1 (Long): Win if >= PT, Loss if <= SL
    # Side -1 (Short): Win if <= PT, Loss if >= SL
    
    is_win = False
    is_loss = False
    
    if not np.isnan(pt_price):
        if side == 1 and exit_price >= pt_price: is_win = True
        elif side == -1 and exit_price <= pt_price: is_win = True
            
    if not np.isnan(sl_price):
        if side == 1 and exit_price <= sl_price: is_loss = True
        elif side == -1 and exit_price >= sl_price: is_loss = True

    if is_win:
        marker = '^' if side == 1 else 'v' # Win usually arrow in direction
        color = 'green'
    elif is_loss:
         marker = 'x'
         color = 'red'
    else:
        marker = 'o' # Time expiry
        color = 'gray'

    plt.scatter(end_time, exit_price, color=color, label='Exit', marker=marker, s=100)
    
    # Extend X-axis if t1 is in the future (Live Mode)
    if end_time > price_path.index[-1]:
        plt.xlim(left=price_path.index[0], right=end_time + pd.Timedelta(minutes=5))
    
    plt.title(f"Trade Box: {start_time} (Vol: {trgt:.4f})")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()
        return None
    elif 'return_buffer' in kwargs and kwargs['return_buffer']:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    else:
        plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plots the Confusion Matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_live_setup(current_price, barriers, history_df, confidence, extra_lines=None):
    """
    Plots the live setup for real-time inference.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot History
    plt.plot(history_df.index, history_df.values, label='History', color='black')
    
    # Current Price
    last_time = history_df.index[-1]
    plt.scatter(last_time, current_price, color='blue', s=100, label='Current Price')
    
    # Projected Barriers (Future)
    # We extend lines from current time to "future" arbitrary points for visualization
    future_time = last_time + pd.Timedelta(minutes=20)
    
    pt = barriers['pt']
    sl = barriers['sl']
    
    plt.hlines(pt, xmin=last_time, xmax=future_time, colors='green', linestyles='--', label='Target (PT)')
    plt.hlines(sl, xmin=last_time, xmax=future_time, colors='red', linestyles='--', label='Stop (SL)')

    # Extra lines (e.g., Breakeven / Entry)
    if extra_lines:
        for label, price in extra_lines.items():
            color = 'yellow' if 'Breakeven' in label or 'Entry' in label else 'orange'
            style = '--'
            plt.hlines(price, xmin=last_time, xmax=future_time, colors=color, linestyles=style, label=label)
    
    plt.title(f"Live Setup (Confidence: {confidence:.2%})")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Annotate Confidence
    plt.text(last_time, current_price, f" {confidence:.2%}", verticalalignment='bottom')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

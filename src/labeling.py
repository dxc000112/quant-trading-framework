import numpy as np
import pandas as pd

def get_daily_vol(close, span=100):
    """
    Computes daily volatility (EWMA of returns).
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    
    # Use .iloc to avoid timezone mismatch issues in index lookup
    past_indices = df0 - 1
    current_indices = np.arange(len(close) - len(df0), len(close))
    
    # Calculate returns (element-wise division using values for denominator)
    daily_returns = close.iloc[current_indices] / close.iloc[past_indices].values - 1
    
    # Result should inherit current index
    return daily_returns.ewm(span=span).std()

def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    """
    Apply stop loss/profit taking, if it takes place before t1 (end of event).
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    
    # Initialize with same dtype (e.g. datetime64[ns, UTC]) to avoid naive/aware conflicts
    out['sl'] = pd.Series(pd.NaT, index=out.index, dtype=out['t1'].dtype)
    out['pt'] = pd.Series(pd.NaT, index=out.index, dtype=out['t1'].dtype)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # Price path
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # Returns
        
        # Stop Loss
        sl_hits = df0[df0 < sl[loc]]
        if not sl_hits.empty:
            out.loc[loc, 'sl'] = sl_hits.index.min()
        else:
             out.loc[loc, 'sl'] = pd.NaT
             
        # Profit Take
        pt_hits = df0[df0 > pt[loc]]
        if not pt_hits.empty:
             out.loc[loc, 'pt'] = pt_hits.index.min()
        else:
             out.loc[loc, 'pt'] = pd.NaT
    return out

def get_events(close, t_events, pt_sl, target_vol, min_ret, vertical_barrier_times=False, side=None):
    """
    Finds the time of the first barrier touch.
    
    Args:
        close: A pandas Series of prices.
        t_events: The pandas DatetimeIndex containing the timestamps that will seed every triple barrier.
        pt_sl: A list of two non-negative float values:
               pt_sl[0] The factor that multiplies trgt to set the width of the upper barrier.
               pt_sl[1] The factor that multiplies trgt to set the width of the lower barrier.
        target_vol: A pandas Series of targets, expressed in terms of absolute returns.
        min_ret: The minimum target return required for running a triple barrier search.
        vertical_barrier_times: A pandas Series with the timestamps of the vertical barriers.
        side: A pandas Series of the side of the bet (long/short).
        
    Returns:
        pd.DataFrame with columns: t1 (earliest touch), trgt (target volatility), side (if provided).
    """
    # 1. Get target
    trgt = target_vol.loc[t_events]
    trgt = trgt[trgt > min_ret]  # Drop trades with low expected return
    
    # 2. Get t1 (Vertical Barrier)
    if vertical_barrier_times is False:
        t1 = pd.Series(pd.NaT, index=t_events)
    else:
        t1 = vertical_barrier_times.loc[trgt.index]
        
    # 3. Form events object
    side_ = side.loc[trgt.index] if side is not None else pd.Series(1., index=trgt.index)
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    
    # 4. Apply Horizontal Barriers
    df0 = apply_pt_sl_on_t1(close, events, pt_sl, events.index)
    
    events['t1'] = df0.dropna(how='all').min(axis=1) # Earliest touch (pt or sl or t1) (min takes earliest date)
    
    # Ensure t1 has the same timezone as the index (if index is tz-aware)
    if events.index.tz is not None:
        if events['t1'].dt.tz is None:
             events['t1'] = events['t1'].dt.tz_localize(events.index.tz)
        else:
             events['t1'] = events['t1'].dt.tz_convert(events.index.tz)

    if vertical_barrier_times is not False:
         events['t1'] = events['t1'].fillna(vertical_barrier_times.loc[events.index])
         
    events['pt'] = df0['pt']
    events['sl'] = df0['sl']
    
    return events.drop('side', axis=1)

def get_bins(events, close):
    """
    Label the events.
    1 if PT touched first.
    0 if SL or Vertical Barrier touched first.
    """
    # 1. Prices aligned with events
    events_ = events.dropna(subset=['t1'])
    
    # Identify unique timestamps involved
    t1_values = events_['t1']
    
    # Ensure both indices are compatible (timezone-aware)
    # Using pd.Index explicitly handles Series -> Index conversion better than .values for TZ
    t1_index = pd.Index(t1_values)
    
    # Union
    px = events_.index.union(t1_index).drop_duplicates()
    px = close.reindex(px, method='bfill')
    
    # 2. Create out object
    out = pd.DataFrame(index=events_.index)
    # Use Series lookup to preserve timezone info and avoid KeyError
    out['ret'] = px.loc[events_['t1']].values / px.loc[events_.index].values - 1
    
    # 3. Meta-Labeling
    # If pt happened before sl, label 1. Else 0.
    # Note: apply_pt_sl_on_t1 returns earliest timestamp.
    # If pt is not NaT and (sl is NaT or pt < sl), then PT was touched first.
    
    # However, standard simpler logic often used:
    # If strictly positive return? No, we should use the barrier logic.
    
    # Let's rely on 'pt' and 'sl' columns if we kept them, but let's re-infer for simplicity or use the return sign if side was 1.
    # Actually, in standard meta-labeling:
    # Label 1 if the primary model *correctly* predicted the direction (i.e. we hit the profit barrier).
    # Label 0 if it failed (hit stop loss or time out).
    
    # We need to pass 'side' to know if return should be positive or negative.
    # But since 'events' dropped side, let's assume we label based on simple profit attainment.
    # If we hit PT, it's a win (1). If we hit SL or T1, it's a loss (0).
    
    def label_row(row):
        if pd.notna(row['pt']):
            if pd.isna(row['sl']) or row['pt'] < row['sl']:
                 return 1
        return 0

    out['bin'] = events.apply(label_row, axis=1)
    
    return out

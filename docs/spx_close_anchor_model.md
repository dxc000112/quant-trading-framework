# SPX Intraday Close Anchor Model

This document deliberately focuses on model system design, data design, factor design, backtest design, live-update logic, and failure modes first. It does not lock in final alpha weights or a trading conclusion.

## A. System Design

### Objective

From 10:00 ET onward, forecast:

- the most likely close-anchor or "magnet" price from now to the cash close
- a forecast interval with upper and lower bounds
- structural option levels that help explain the anchor: `target_price`, `long_gamma`, `call_wall`, `put_wall`, `gex_pain`, `gamma_ranks`
- a refreshed markdown card every 5 minutes

### Architectural Split

The model should be treated as six cooperating layers instead of one monolithic strategy:

1. `Spot Layer`
   Ingest 1-minute SPX cash or a tightly aligned proxy with OHLCV and intraday VWAP.

2. `Options Structure Layer`
   Ingest same-day and near-date SPX/SPXW option snapshots every 5 minutes, normalize strikes, expiry, Greeks, open interest, and volume, then derive structure levels such as call wall, put wall, long gamma, and gamma balance.

3. `Feature Layer`
   Join early-session price behavior with option structure distances, concentration, and regime descriptors into a snapshot feature row.

4. `Label Layer`
   Convert the realized future intraday path from each decision time to a backtestable target:
   `target_price`, `lower_bound`, and `upper_bound`.

5. `Forecast Layer`
   Train a point model for `target_price` and quantile models for interval bounds. Keep this trainable rather than hard-coding final weights too early.

6. `Publishing Layer`
   Render each forecast into a readable markdown summary card and track update cadence, freshness, and fallback status.

### Recommended Module Layout

The implementation in this repo is organized as:

- `src/spx_anchor/config.py`
- `src/spx_anchor/structure.py`
- `src/spx_anchor/features.py`
- `src/spx_anchor/targets.py`
- `src/spx_anchor/model.py`
- `src/spx_anchor/backtest.py`
- `src/spx_anchor/live.py`
- `src/spx_anchor/reporting.py`

This keeps the data vendor problem separate from the modeling problem.

## B. Data Requirements

### Required Market Data

1. `SPX` 1-minute bars
   Required fields: `Open`, `High`, `Low`, `Close`, `Volume`

2. Option-chain snapshots every 5 minutes
   Required fields:
   - `as_of`
   - `expiry`
   - `option_type`
   - `strike`
   - `gamma`
   - `open_interest`
   - `volume`

3. Preferred optional fields
   - `iv`
   - `delta`
   - `bid`
   - `ask`
   - `mid`

### Data Frequency

- Spot bars: 1 minute
- Structure refresh: every 5 minutes
- First publish time: 10:00 ET
- Last publish time: usually 15:55 ET or the final available update before close

### Important Data Notes

- Intraday open interest is stale by construction. For 0DTE, same-day `volume` must be blended with `open_interest` to avoid over-trusting yesterday's open interest map.
- If live Greeks are unavailable, gamma must be computed upstream from IV and contract specs before entering this model.
- SPX index and option timestamps must be aligned to the same session clock. Small timestamp mismatches will create fake wall shifts.
- Near-event days need event flags if possible: CPI, FOMC, OPEX, monthly expiry, quarterly expiry, holiday half days.

## C. Factor Definitions

### 1. Price and Session Factors

- `opening_drive_pct`
  `close_at_10:00 / open_at_09:30 - 1`

- `opening_range_pct`
  `(high_09:30_10:00 - low_09:30_10:00) / close_at_10:00`

- `drive_efficiency`
  `abs(close_at_10:00 - open_at_09:30) / max(initial_balance_width, epsilon)`

- `vwap_distance`
  `spot_now - session_vwap_now`

- `ib_mid_distance`
  `spot_now - initial_balance_mid`

- `return_since_anchor_start`
  `spot_now / close_at_10:00 - 1`

- `realized_vol_5m`, `realized_vol_30m`
  intraday realized volatility estimates from 1-minute returns

- `overnight_gap_pct`
  `open_at_09:30 / previous_close - 1`

### 2. Structure Factors

- `call_wall`
  Strike with the largest call-side gamma or call-side size concentration.

- `put_wall`
  Strike with the largest put-side gamma or put-side size concentration.

- `long_gamma`
  Strike with the strongest positive net gamma concentration under the model's sign convention.

- `gex_pain`
  Gamma balance point where cumulative signed gamma on both sides of price is most balanced.

- `gamma_flip`
  Nearest strike interval where signed net gamma changes sign.

- `wall_width`
  `call_wall - put_wall`

- `dist_call_wall`, `dist_put_wall`, `dist_long_gamma`, `dist_gex_pain`
  Signed point distance from spot to each structure level

- `spot_inside_walls`
  Indicator that spot lies between `put_wall` and `call_wall`

### 3. Gamma Regime and Concentration Factors

- `total_gex`
  Total signed gamma exposure across the included expiries

- `front_expiry_gex`
  Total signed gamma exposure for the nearest expiry bucket

- `gamma_slope`
  Local slope of net gamma by strike around spot

- `call_concentration`
  Share of gross call gamma concentrated at the largest call strike

- `put_concentration`
  Share of gross put gamma concentrated at the largest put strike

- `gamma_ranks`
  Historical percentile ranks of:
  - `total_gex`
  - `front_expiry_gex`
  - `gamma_slope`
  - `call_concentration`
  - `put_concentration`

### 4. Label Definitions

- `target_price`
  The weighted mode of future 1-minute close prices from decision time to cash close, rounded to a configurable price bucket.

- `lower_bound`, `upper_bound`
  Weighted future-price quantiles, for example the 15th and 85th percentiles.

This turns "magnet" into a concrete supervised-learning target instead of a narrative-only concept.

## D. Backtest Plan

### Unit of Prediction

Each row is one 5-minute decision time from 10:00 ET to the close for one session.

### Training / Testing Protocol

- build snapshot rows session by session
- use expanding walk-forward retraining
- retrain every N sessions
- never let future rows from the same date leak into rank features or model fitting

### Baselines

Always compare the model against simple baselines:

- current spot
- session VWAP
- `gex_pain`
- midpoint of call wall and put wall

### Core Metrics

- `target_mae`
- `target_rmse`
- interval coverage
- average interval width
- hit rate within 5 points
- hit rate within 10 points
- error vs baseline anchors
- stability of the predicted anchor across the day

### Slice Analysis

Break out results by:

- positive gamma vs negative gamma sessions
- high event-risk days
- large overnight gap days
- monthly OPEX / quarterly expiry
- first prediction at 10:00 vs later refreshes

## E. Live Update Logic

### Timeline

1. `09:25-09:30`
   Warm cache, prior close, prior ranks, and pre-open structure files.

2. `09:30-10:00`
   Accumulate opening-drive and initial-balance features.

3. `10:00`
   Publish first close-anchor forecast.

4. `Every 5 minutes`
   Refresh bars and latest option snapshot, recompute structure, feature row, prediction interval, and markdown card.

### Live Rules

- only use the latest same-session option snapshot that is not stale
- if option data is stale, downgrade confidence or hold the prior forecast with a stale-data warning
- keep rank features sourced only from history before the current update timestamp
- if a new update changes target by less than a small threshold, mark it as "stable"
- if call wall / put wall / gamma regime shifts materially, mark it as "regime change"

### Output Card

The card should include:

- spot price and update time
- target price
- forecast interval
- long gamma
- call wall
- put wall
- gex pain
- gamma ranks
- a short note about regime and primary drivers

## F. Risks and Failure Scenarios

### Structural Risks

- 0DTE open interest is stale and can misrepresent same-day positioning
- dealer sign assumptions can be wrong in single-name or event-driven flow, and less wrong but still imperfect in SPX
- a large macro event can overwhelm structure-based pinning logic

### Data Risks

- missing strikes or truncated chains will move walls artificially
- delayed option snapshots will create fake persistence
- using a proxy like SPY or ES for spot introduces basis risk vs SPX

### Model Risks

- if the target label is too wide or too smooth, the model will collapse to current spot
- if the target bin size is too small, noise will dominate and the anchor becomes unstable
- if interval models are under-trained, they will be badly calibrated on event days

### Operational Risks

- half-days and holiday sessions need separate session-close handling
- wall levels may jump after large directional moves if strikes roll in and out of relevance
- if structure data drops but spot remains live, the card can look precise while actually stale

## Coding Notes

The first implementation in this repo should:

- keep the forecasting layer trainable
- keep vendor-specific fetching abstract
- define targets and structure consistently for both backtest and live
- prefer transparency over complexity in the first version

That is the right order for this model. We want a robust measurement and publishing system before we argue about the final predictive weights.

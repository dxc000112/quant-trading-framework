# SPX / SPXW Close-Anchor Research Memo

This memo is intentionally research-first. It defines the data map, output semantics, baseline models, dealer-positioning proxies, bias map, label design, and intraday rolling-update logic for an SPX / SPXW close-anchor model. It is not the final production-system spec.

## 1. Full Data-Source Map

The model can be built in tiers. The first tier is mandatory for any useful baseline. The second tier materially improves calibration. The third tier is optional but important around event days.

### A. Core Underlying Data

1. `SPX` cash index minute bars
   Fields:
   - timestamp
   - open, high, low, close
   - if available: disseminated index value updates or vendor minute aggregates

   Purpose:
   - anchor target is defined on SPX cash close
   - compute opening drive, first-30m range, VWAP distance, realized vol, and time-to-close features

2. `ES` front-month futures minute bars
   Fields:
   - timestamp
   - open, high, low, close, volume
   - if available: session VWAP

   Purpose:
   - faster directional proxy than SPX cash
   - detect SPX-ES divergence, futures-led repricing, and shock continuation vs mean reversion

3. Trading calendar and special-session calendar
   Fields:
   - normal session close
   - half-day flag
   - holiday / early-close status

   Purpose:
   - define the true prediction horizon
   - avoid incorrect time-to-close and end-of-day label leakage

### B. Core Option-Structure Data

4. `SPX/SPXW` option-chain snapshots
   Granularity:
   - ideally every 1 minute
   - minimum acceptable for the baseline: every 5 minutes

   Fields:
   - as_of
   - expiry
   - strike
   - call_or_put
   - open_interest
   - volume
   - bid, ask, mid if available
   - implied_volatility
   - greeks if vendor supplies them

   Purpose:
   - compute call wall, put wall, long gamma level, weighted gamma center, max pain, pin-risk level, strike-level attractor ranks

5. End-of-day open-interest history
   Fields:
   - trade_date
   - expiry
   - strike
   - call_or_put
   - official OI

   Purpose:
   - baseline inventory proxy
   - compare current volume with stale OI
   - estimate whether same-day flow has likely displaced prior positioning

### C. Greeks / Surface Inputs

6. Vendor-supplied Greeks
   Preferred fields:
   - delta
   - gamma
   - vanna
   - charm
   - theta

   Purpose:
   - direct structure estimation
   - detect whether intraday hedging pressure should weaken or strengthen pinning

7. If Greeks are not supplied: local Greek engine inputs
   Required inputs:
   - underlying spot or forward
   - implied vol or IV surface
   - time to expiry
   - interest rate
   - dividend / carry assumption

   Purpose:
   - compute gamma, delta, vanna, charm locally

### D. Event / News / Regime Data

8. Macro-event calendar
   Fields:
   - CPI
   - FOMC
   - NFP
   - OPEX
   - quarter-end
   - month-end
   - holiday-adjacent sessions

   Purpose:
   - mark days where intraday structure is frequently overridden or repriced

9. Headline / shock feed
   Minimal requirement:
   - a timestamped shock flag

   Better version:
   - timestamped headlines
   - simple severity / category tag

   Purpose:
   - trigger shock-repricing state
   - widen forecast interval and reduce reliance on static OI-based structure

### E. Optional Calibration Data

10. `SPY` / `XSP` / related ETF-option data
    Purpose:
    - cross-check retail-heavy vs institutional-heavy flow proxies
    - compare whether same-day flow in smaller products hints at directional crowding

11. `VIX`, `VVIX`, short-dated vol-index series
    Purpose:
    - detect vol-of-vol shocks
    - adjust expected move and confidence score

12. Intraday options trade prints or trade classification
    Purpose:
    - better sign same-day opening flow
    - partially replace stale OI assumptions

### Minimal Viable Data Stack

For the first research-quality baseline, the absolute minimum is:

- SPX minute bars
- ES minute bars
- SPX/SPXW option snapshots every 5 minutes
- official end-of-day OI
- IV or vendor gamma
- event calendar

Without that set, the model quickly degenerates into a narrative overlay rather than a measurable forecast engine.

## 2. Mathematical Meaning of Each Output Field

Let:

- `t` = current intraday update time
- `T_close` = cash close
- `S_t` = spot at time `t`
- `K` = strike
- `A_t(K)` = strike-level attractor score
- `G(K)` = signed gamma exposure proxy at strike `K`
- `|G|(K)` = gross gamma exposure proxy

### A. Forecast Outputs

1. `target_price`

Mathematical meaning:

`target_price_t = argmax_p P(Close_anchor in bucket p | X_t)`

Interpretation:

- not the expected market direction
- not the most likely next 5-minute move
- the model's best estimate of the price bucket toward which the intraday path is most likely to converge from `t` to the close

2. `forecast_low`

Mathematical meaning:

`forecast_low_t = Q_alpha(Close_anchor | X_t)`

Usually `alpha` should start around `0.15` or `0.20`.

3. `forecast_high`

Mathematical meaning:

`forecast_high_t = Q_(1-alpha)(Close_anchor | X_t)`

Interpretation of the interval:

- this is a conditional convergence band, not simply realized volatility bands
- the interval should widen on negative-gamma or shock-repricing states

### B. Structural Outputs

4. `long_gamma_level`

Define signed gamma proxy at strike:

`G(K) = sum_i sign_i * gamma_i * size_i * multiplier * S_t^2`

where `sign_i` is determined by the dealer-position proxy assumption and `size_i` is an adjusted size proxy.

Then:

`long_gamma_level_t = argmax_K G(K)` subject to `G(K) > 0`

5. `call_wall`

`call_wall_t = argmax_K CallGrossGamma(K)`

or, if Greek quality is weak,

`call_wall_t = argmax_K CallSizeConcentration(K)`

6. `put_wall`

`put_wall_t = argmax_K PutGrossGamma(K)`

or its size-concentration proxy.

7. `gex_pain`

Define:

`CumLeft(p) = sum_(K <= p) G(K)`

`CumRight(p) = sum_(K > p) G(K)`

Then:

`gex_pain_t = argmin_p |CumLeft(p) - CumRight(p)|`

Interpretation:

- a gamma-balance center
- not the same thing as classical option max pain

8. `gamma_rank_1 ... gamma_rank_5`

First define strike-level attractor score:

`A_t(K) = w1 * GrossGammaShare(K) + w2 * SizeShare(K) + w3 * PinRisk(K) + w4 * DistanceScore(K)`

with:

- `GrossGammaShare(K) = |G|(K) / sum_j |G|(j)`
- `SizeShare(K) = Size(K) / sum_j Size(j)`
- `PinRisk(K) = min(CallSize(K), PutSize(K)) / max(Size(K), epsilon)`
- `DistanceScore(K) = 1 / (1 + ((K - S_t) / EM_t)^2 )`

Then `gamma_rank_n` is the `n`-th highest strike under `A_t(K)`.

Interpretation:

- these are the top candidate magnetic strikes
- they are more informative than a single net-GEX scalar

9. `market_regime`

Define:

`RegimeRatio_t = TotalSignedGEX_t / TotalGrossGEX_t`

Then:

- `long_gamma` if `RegimeRatio_t >= theta_pos`
- `short_gamma` if `RegimeRatio_t <= theta_neg`
- `mixed` otherwise

Recommended initial thresholds:

- `theta_pos = +0.15`
- `theta_neg = -0.15`

10. `confidence_score`

This should not be a pure model probability.

A more defensible definition is:

`confidence_t = f(consensus, concentration, regime_stability, data_freshness, shock_state, time_of_day)`

where:

- higher baseline agreement raises confidence
- stronger strike concentration raises confidence
- stale chains lower confidence
- shock repricing lowers confidence
- long-gamma late-session states often deserve higher confidence than short-gamma early-session states

11. `invalidation_conditions`

This is a set of rule-based failure triggers, not a number.

Examples:

- two consecutive 5-minute closes outside the forecast band
- spot breaks wall levels by more than `x` points
- regime flips from long-gamma to short-gamma
- shock flag turns on
- fresh option snapshot materially relocates rank-1 and rank-2 attractor strikes

## 3. Baseline Design

The baselines should be explicit, interpretable, and individually measurable.

### Baseline 1: Net GEX + Call/Put Wall

Purpose:

- simplest structure-aware anchor
- captures the idea that price tends to hover near the dominant dealer-hedging zone

Suggested formula:

`B1_t = a1 * gex_pain_t + a2 * long_gamma_level_t + a3 * mid(call_wall_t, put_wall_t) + a4 * nearest_wall_t`

with regime-dependent weights:

- in `long_gamma`: put more weight on `gex_pain`, `long_gamma_level`, and wall midpoint
- in `short_gamma`: move some weight back toward live spot and nearest wall

Strength:

- transparent

Weakness:

- can become too static on shock days

### Baseline 2: Max Pain / Pin Risk

Purpose:

- capture expiry-day pinning around strikes where both call and put inventory are meaningful

Required components:

- classical `max_pain`
- `pin_risk_level = argmax_K PinRisk(K)`

Suggested formula:

`B2_t = b1 * max_pain_t + b2 * pin_risk_level_t + b3 * initial_balance_mid_t`

Interpretation:

- max pain is slow-moving
- pin risk responds more locally to concentrated two-sided size at a strike

Strength:

- especially useful on same-day expiry and late session

Weakness:

- classical max pain often overweights stale OI

### Baseline 3: Intraday Mean Reversion to Weighted Gamma Center

Define weighted gamma center:

`WGC_t = sum_K K * A_t(K) / sum_K A_t(K)`

Then:

`B3_t = S_t + rho_t * (WGC_t - S_t)`

where `rho_t` depends on:

- market regime
- time to close
- shock state

Suggested behavior:

- larger `rho_t` in long-gamma late session
- smaller `rho_t` in short-gamma or shock state

Strength:

- adapts better than static walls

Weakness:

- still depends on proxy positioning

## 4. Estimating Dealer Positioning Without the Real Dealer Book

This is the hardest part. We do not observe the actual dealer inventory, so we need a proxy stack rather than a single estimate.

### A. Core Assumption

For index options, the usual starting point is:

- public / institutional customers are net long convexity at many important strikes
- dealers are often the other side

That implies a first-pass sign convention:

- call gamma contributes positive dealer gamma
- put gamma contributes negative dealer gamma

But this is only a proxy, not truth.

### B. Size Proxy

Use adjusted size:

`Size_i,t = OI_i,t + lambda * Volume_i,t`

with `lambda` between `0` and `1`.

Interpretation:

- OI is stale but persistent
- same-day volume helps capture fresh positioning

### C. Signed Greek Proxies

For contract `i`:

- `GEX_i,t = sign_i * Gamma_i,t * Size_i,t * Multiplier * S_t^2`
- `DEX_i,t = sign_i * Delta_i,t * Size_i,t * Multiplier * S_t`
- `VEX_i,t = sign_i * Vanna_i,t * Size_i,t`
- `CEX_i,t = sign_i * Charm_i,t * Size_i,t`

Use them as relative intraday pressure measures rather than pretending they are exact dealer books.

### D. Expiry-Bucket View

Estimate positioning separately for:

- `0DTE`
- `1DTE`
- `weekly`
- `standard monthly`

Reason:

- 0DTE flow is path-dependent and can overwhelm slower monthly positioning intraday

### E. Composite Dealer-Pressure Proxy

One useful composite is:

`DealerPressure_t = c1 * NetGEX_t + c2 * NetVanna_t + c3 * NetCharm_t`

with time-dependent coefficients:

- charm matters more as time decays into the close
- vanna matters more when spot-vol correlation is active

### F. Practical Research Rule

Never trust one sign convention blindly.

Backtest at least three variants:

1. `dealer short customer` default sign
2. `reduced-sign-confidence` shrinkage toward zero
3. `flow-adjusted` sign where same-day volume near the offer or bid, if available, tilts the sign

## 5. Proxy Variables vs Serious Bias Risks

### A. Things That Are Necessarily Proxies

1. `dealer positioning`
   We do not observe the real dealer book.

2. `intraday effective inventory`
   OI is yesterday's inventory, not today's current state.

3. `same-day opening flow sign`
   Without trade classification, same-day volume direction is ambiguous.

4. `vanna` and `charm` pressure
   Even if the Greeks are computed correctly, the sign of the dealer inventory is still inferred.

5. `headline shock`
   A news flag is only a partial proxy for how positioning has changed after the headline.

### B. Where Serious Bias Can Arise

1. Stale OI on 0DTE
   Biggest source of structural error.

2. Missing strikes or truncated chains
   Artificially shifts call wall, put wall, rank-1 magnet, and max pain.

3. Misaligned timestamps
   If SPX and options snapshots are even a few minutes off, wall changes can be fake.

4. Using SPY or ES as a substitute for SPX cash without basis correction
   Basis can matter materially around macro moves and into settlement windows.

5. Assuming all index-option flow maps to one dealer-sign convention
   This can flip the interpretation of net gamma.

6. Treating max pain as an executable magnetic truth
   It is often too slow and too stale intraday.

7. Ignoring event-state transitions
   A CPI or FOMC release can invalidate all pre-event pinning logic.

### C. Bias Severity Ranking

Highest severity:

- stale OI
- wrong sign convention
- missing chain coverage
- timestamp mismatch

Medium severity:

- poor IV / Greek quality
- ES-SPX basis mismatch
- weak headline classification

Lower severity:

- exact confidence-score formula
- exact wall-weight coefficients

## 6. Backtest Label Definition

The label must reflect convergence, not direction.

### A. Decision Times

Create one row every 5 minutes from `10:00 ET` to the final update before the close.

Each row sees only information available at that timestamp.

### B. Future Window

For decision time `t`, define the future path:

`P_t = { S_u : u in [t, T_close] }`

### C. Target Label

Define a price bucket size, initially `5` points.

Then:

`label_target_price_t = argmax_p WeightedCount(S_u in bucket p for u in [t, T_close])`

This is the weighted future-price mode.

Suggested weights:

- uniform at first
- then experiment with volume weights or recency weights

### D. Interval Labels

`label_forecast_low_t = Q_0.15(P_t)`

`label_forecast_high_t = Q_0.85(P_t)`

These are path-quantile labels, not realized-high / realized-low labels.

### E. Auxiliary Labels

Also store:

- future close
- future VWAP
- whether final close landed within 5 points of rank-1 attractor
- whether forecast regime flipped before close

These auxiliary labels help explain why the forecast worked or failed.

### F. Metrics

Measure:

- MAE of `target_price`
- RMSE of `target_price`
- interval coverage
- interval width
- hit rate within 5 / 10 points
- stability across updates
- improvement over each baseline

## 7. Intraday Rolling-Update Logic

The model should treat the session in phases.

### Phase 0: Pre-Open Preparation

Time:

- roughly `09:00-09:29 ET`

Tasks:

- load prior close
- load official OI history
- warm latest option files
- build event-day flags
- initialize prior-day rank distributions

### Phase 1: Opening Discovery Window

Time:

- `09:30-10:00 ET`

Rules:

- do not trust early close-anchor forecasts too much
- accumulate first-30m range, VWAP, opening drive, and early structure migration
- track whether walls are stable or still moving

Output:

- optional provisional note
- no full-confidence anchor unless explicitly desired

### Phase 2: First Forecast

Time:

- `10:00 ET`

Tasks:

- freeze first-30m features
- compute strike-level concentration map
- estimate dealer-pressure proxies
- emit the first full forecast

### Phase 3: Normal Rolling Updates

Time:

- every 5 minutes from `10:05 ET` onward

At each update:

1. ingest the newest SPX / ES bars
2. ingest the freshest option-chain snapshot
3. check staleness and chain completeness
4. recompute structure levels and rank-1 to rank-5 attractor strikes
5. recompute all three baselines
6. combine or compare them
7. update `target_price`, interval, confidence, and invalidation conditions

### Phase 4: Shock Re-Estimation

Trigger:

- headline shock
- abnormal 5-minute return
- realized-vol jump
- wall relocation beyond threshold

Rules:

- reduce the weight on stale OI-heavy baselines
- increase weight on live spot, ES, and freshest structure
- widen interval
- lower confidence
- annotate output as a shock-repriced forecast

### Phase 5: Late-Day Convergence Logic

Time:

- roughly last `60-90` minutes before close

Rules:

- long-gamma sessions can justify stronger convergence toward top-ranked attractor levels
- short-gamma sessions should not be forced into false pin assumptions
- if wall structure remains stable while spot oscillates around rank-1 / rank-2 levels, confidence may rise

### Live Invalidation Workflow

At each 5-minute update, mark the prior forecast invalid if:

- the band is broken twice in a row
- wall levels are materially redrawn
- regime flips
- a shock flag is raised

Then immediately re-estimate rather than carrying forward a stale anchor.

## Recommended Research Order

1. validate data integrity and time alignment
2. measure Baseline 1, 2, and 3 separately
3. test regime segmentation: long-gamma / short-gamma / mixed
4. test event-day slices separately
5. only then fit a combined predictive model

That order matters. If the baselines are not stable and interpretable, a more complex model will only hide the data and sign-assumption errors rather than solve them.

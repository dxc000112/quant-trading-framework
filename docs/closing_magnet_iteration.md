# Closing Magnet Iteration

## Scope
This iteration tightens the `V3` engine into a closing magnet model, not a direction model.

- Observable:
  - `SPX` spot path, VWAP distance, realized vol, call wall, put wall, listed strike gamma/OI concentration, time to close.
- Approximated:
  - Dealer positioning via `OI + lambda * intraday volume`.
  - Headline shock severity via flags plus de-anchor / vol-spike proxies.
- Inferred:
  - `shock_score`
  - `pin_trust_multiplier`
  - `event_shock / trend_day / pin_day` regime choice
  - final `target_price`

## New Factor: Shock Filter

### Intuition
Gamma pin logic is most fragile right after the open and during macro/headline dislocations. The model should trust pinning less in those windows and only restore that weight after realized vol and structure stabilize.

### Formula
Let:

- `deanchor_score = clip(max(|spot - magnet_price|, |spot - pin_strike|, outside_wall_distance, 0.75 * |vwap_distance|) / range_scale - 0.55, 0, 1)`
- `vol_spike_score = clip((rv_5 / rv_30 - 1.15) / 0.85, 0, 1)`
- `structure_break_score = clip(|pin_strike - magnet_price| / range_scale - 0.25, 0, 1)`
- `shock_score = 0.35 * event_score + 0.30 * deanchor_score + 0.20 * vol_spike_score + 0.15 * structure_break_score`
- `opening_trust_cap = 0.35` before 10:00 ET, then linearly ramps to `1.0`
- `pin_trust_multiplier = opening_trust_cap * (1 - 0.85 * shock_score) * (0.55 + 0.45 * recovery_score)`

### Expected Benefit
- Prevents late-day pin logic from being over-trusted on `FOMC / CPI / headline` dislocations.
- Keeps the first 30 minutes explicitly separate from the balance phase.
- Lets pin weight recover when vol converges and structure re-forms.

### Failure Mode
- Event flags can miss unscheduled flows.
- Realized vol can stay low during a slow grind trend, which may overstate trust.
- With no true dealer book, this remains a positioning proxy rather than an observed inventory measure.

## V3 Target Logic

### Intuition
Every regime should still point to a structural closing magnet basket. Regime only changes how much trust is assigned to `pin_anchor` versus broader structure, not whether the model extrapolates direction.

### Formula
Structural candidates:

- `magnet_core = wavg(weighted_gamma_magnet_price, strongest_magnet_strike, weighted_gamma_center, rank_target)`
- `wall_anchor = wavg(v2_target, nearest_wall, max_pain, session_vwap)`
- `pin_anchor = wavg(strongest_pin_strike, strongest_magnet_strike, max_pain, nearest_wall)`

Regime-specific target:

- `target = blend(magnet_core, wall_anchor, regime_wall_weight)`
- `target = blend(target, pin_anchor, effective_pin_weight)`
- `effective_pin_weight = regime_pin_weight * pin_trust_multiplier * f(strongest_pin_score)`

No regime uses a drift projection term.

### Expected Benefit
- Preserves interpretability.
- Keeps `trend_day` and `short_gamma` outputs as closing anchor estimates rather than directional forecasts.
- Makes target shifts traceable to changing structure instead of opaque momentum projection.

### Failure Mode
- On true news repricing days, even broad structure anchors may lag the new equilibrium.
- If option snapshots are stale or missing strikes, the structural basket can become biased.

## Verification
- Code updated in `src/spx_anchor/features.py`, `src/spx_anchor/versioned.py`, `src/spx_anchor/reporting.py`, and `src/spx_anchor/types.py`.
- Tests added in `tests/test_spx_anchor_versions.py`.
- Backtest comparison should be regenerated from the synthetic research scaffold before promoting further changes.

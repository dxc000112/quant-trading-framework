# AGENTS.md

## Project goal
Build an interpretable intraday SPX closing magnet model based on options positioning and intraday market structure.

## Working rules
- Never jump straight to final modeling without baselines.
- Always explain assumptions around dealer positioning approximation.
- Prefer simple interpretable factors before black-box models.
- Keep the first 30 minutes separate from the rest of the session.
- Treat event days and normal pin days as different regimes.
- Whenever data or assumptions are insufficient, do not hallucinate precision.
- State clearly what is observable, what is approximated, and what is inferred.
- Every new factor must include:
  1. intuition
  2. formula
  3. expected benefit
  4. failure mode
- Every model iteration must include:
  1. code
  2. test
  3. backtest comparison
  4. markdown summary

## Output format
The live report must always include:
- target_price
- forecast range
- long_gamma
- call wall
- put wall
- gex pain
- top 5 gamma ranks
- regime
- confidence
- invalidation logic

## Coding rules
- Use Python
- Use pandas/polars for data wrangling
- Keep factor calculations modular
- Write docstrings for every factor
- Add unit tests for core calculations

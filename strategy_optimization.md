# Strategy Optimization Report (Jan 12-16, 2026)

## Static Thresholds (Precision-Recall)
|   Threshold |   Signals |   Win Rate |   Wins |   Losses |
|------------:|----------:|-----------:|-------:|---------:|
|        0.20 |    883.00 |       0.58 | 513.00 |   286.00 |
|        0.25 |    717.00 |       0.56 | 404.00 |   243.00 |
|        0.30 |    586.00 |       0.55 | 322.00 |   208.00 |
|        0.35 |    470.00 |       0.60 | 281.00 |   146.00 |
|        0.40 |    347.00 |       0.65 | 225.00 |   102.00 |
|        0.45 |    174.00 |       0.67 | 117.00 |    52.00 |
|        0.50 |     56.00 |       0.61 |  34.00 |    20.00 |
|        0.55 |     19.00 |       0.42 |   8.00 |    11.00 |
|        0.60 |      3.00 |       0.00 |   0.00 |     3.00 |
|        0.65 |      0.00 |       0.00 |   0.00 |     0.00 |
|        0.70 |      0.00 |       0.00 |   0.00 |     0.00 |
|        0.75 |      0.00 |       0.00 |   0.00 |     0.00 |
|        0.80 |      0.00 |       0.00 |   0.00 |     0.00 |

## Dynamic Threshold Strategy
**Logic**:
- Volatility Rank > 80% (High Vol): Threshold = 0.65
- Volatility Rank < 20% (Low Vol): Threshold = 0.40
- Normal Regime: Threshold = 0.50

**Performance**:
- Signals: 148
- Win Rate: 70.27%
- Compare to Static 0.50: 60.71%

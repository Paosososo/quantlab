<!-- Snapshot of the 2026-09-27 FRED SP500 pilot. The raw CSV and full JSON results
     are retained under data/artifacts/real-market-pilot-2026-09-27/ (gitignored). -->

# Real-market forecast pilot: S&P 500

This is a close-only forecast experiment, not a trading backtest.
Source: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500
Raw CSV SHA-256: `7980ba09dccc3cc67825fe5355a3fdf2347d0af039fc0336c870e9e0c2681fe3`
Observed closes: 2016-09-26 to 2026-09-25
Valid closes: 2514; model rows: 2492.

The raw CSV and full machine-readable results are local gitignored artifacts,
not part of this repository. The script in `scripts/run_real_market_pilot.py`
can rerun the method on a new FRED download, but the data may have changed and
the numbers may differ from this snapshot.

Five chronological expanding folds, 252 test sessions each, with a two-day embargo.
The target is the next observed session's close-to-close price return.
Features use the current and earlier closes only.

## Forecast accuracy

| model             |    n |     rmse |   r2_oos_vs_zero |   directional_accuracy |   information_coefficient |
|:------------------|-----:|---------:|-----------------:|-----------------------:|--------------------------:|
| zero              | 1260 | 0.010732 |         0        |             nan        |                nan        |
| historical_mean   | 1260 | 0.010723 |         0.001619 |               0.53254  |                 -0.037597 |
| last_value        | 1260 | 0.015304 |        -1.03348  |               0.500794 |                 -0.018703 |
| ridge             | 1260 | 0.010938 |        -0.038811 |               0.502381 |                  0.002406 |
| elastic_net       | 1260 | 0.010881 |        -0.028036 |               0.503175 |                  0.006799 |
| random_forest     | 1260 | 0.010751 |        -0.003553 |               0.515873 |                  0.017507 |
| gradient_boosting | 1260 | 0.011019 |        -0.054255 |               0.509524 |                  0.010049 |

## Statistical comparison with a zero-return forecast

Paired squared-loss differences are ordered by date. The test uses HAC standard
errors and Benjamini-Hochberg adjusted p-values across the six comparisons.
A negative statistic favours the candidate model.

| model             |   n_dates |   hac_lags |   mean_loss_differential |   dm_statistic |   p_value | favours   |   adjusted_p_value | beats_zero_at_5pct   |
|:------------------|----------:|-----------:|-------------------------:|---------------:|----------:|:----------|-------------------:|:---------------------|
| historical_mean   |      1260 |          7 |                -1.87e-07 |        -0.5739 |  0.566    | a         |           0.655    | False                |
| last_value        |      1260 |          7 |                 0.000119 |         6.0978 |  1.43e-09 | b         |           8.57e-09 | False                |
| ridge             |      1260 |          7 |                 4.47e-06 |         2.5547 |  0.0107   | b         |           0.0322   | False                |
| elastic_net       |      1260 |          7 |                 3.23e-06 |         2.0243 |  0.0432   | b         |           0.0647   | False                |
| random_forest     |      1260 |          7 |                 4.09e-07 |         0.4467 |  0.655    | b         |           0.655    | False                |
| gradient_boosting |      1260 |          7 |                 6.25e-06 |         2.3892 |  0.017    | b         |           0.0341   | False                |

## Limits

- Index levels are price-only and omit dividends.
- FRED supplies closes only; no OHLCV execution backtest or trading-cost estimate was run.
- The 22:00 UTC session label is not a verified historical FRED publication timestamp.
- One index and one historical sample do not establish a persistent market edge.

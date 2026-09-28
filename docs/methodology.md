# Methodology

## Research question

> Do machine-learning models provide statistically and economically meaningful
> improvements over simple statistical baselines for financial time-series
> prediction, after realistic transaction costs?

Three words carry the weight. **Statistically**: is the improvement larger than
sampling noise? **Economically**: does it survive costs and turn into money?
**After realistic transaction costs**: measured under frictions someone could
actually have paid.

A negative answer is a valid answer, and the study is built so that reporting
one costs nothing. There is no step that picks the best model and reports only
that.

## Biases, and what is done about each

### Look-ahead bias

*The problem.* Using information that was not available at the decision time.

*What is done.* Four layers, described in
[`backtesting.md`](backtesting.md#four-independent-controls): the `available_at`
column, the point-in-time view, the execution guard, and database CHECK
constraints. Each operates at a different layer, so a bug must defeat all four.

*Residual risk.* The `available_at` values are derived, not observed. Prices use
the regular-hours close, ignoring early closes — a few hours late on a handful of
days per year, which errs conservative. Macro uses a per-series publication lag
set at the slow end of the historical range.

### Data leakage in features

*The problem.* A transform that reads forward — a centred window, a backfill, a
whole-sample scaler.

*What is done.* Two independent controls.

The **structural** one is `PointInTimeFrame`: code is handed an object that
physically contains no future rows.

The **behavioural** one is the leakage harness. It computes features on the full
history, then corrupts everything after a cut point and recomputes. If any value
at or before the cut moved, the feature read the future. It makes no assumption
about *how* a feature is implemented, so it catches leaks a code review misses.

Eight deliberately-broken feature functions are checked into
`tests/leakage/test_leakage_harness.py` and the harness must catch every one:
centred rolling, whole-sample z-score, backfill, time interpolation, a shifted
target, min-max scaling on the full sample, a reversed rolling window and a
reversed expanding mean. Four correct functions must pass. The production feature
library is checked on every CI run.

> **A finding worth recording.** An earlier version of the harness corrupted the
> future with values a thousand times the series' scale, and produced false
> positives on `rolling(...).skew()`. The cause turned out to be pandas: its
> rolling `skew`, `kurt` and `var` subtract a **globally** computed constant from
> the whole array before the sliding computation, for numerical stability.
> Changing the tail by three orders of magnitude changes that constant, which
> changes the rounding of head values. The head was not reading the future; the
> library was re-centring the array. The corruption model now stays at the
> series' own scale, and the artefact is reproduced in a test so the reasoning is
> checked rather than merely asserted in a comment.

*Residual risk.* The harness proves a feature does not read the future *of its
own input frame*. A leak introduced upstream — a provider that silently
back-adjusts history — would not be caught here. The raw layer exists partly so
that such a change is detectable by comparing archived payloads.

### Incorrect train/test boundaries

*The problem.* Random splitting on time series is interpolation, not forecasting.
Even with chronological splits, an `h`-day forward return near a fold boundary is
realised *inside* the test window.

*What is done.* No shuffling option exists anywhere in
`quantlab.models.splitters` — not as a default, not as a flag. Splits are defined
over unique **timestamps**, so a panel of several symbols cuts cleanly.

**Purging** drops every training row whose label became knowable at or after the
test window opens. This is why `build_labels` carries `label_available_at`:
without it, purging would have to guess the horizon.

**Embargo** drops a further gap, because serial correlation means observations
immediately before the test window carry nearly the same information as those
inside it. (Both from López de Prado, *Advances in Financial Machine Learning*,
ch. 7.)

A test asserts that a splitter has no `shuffle` attribute, as a guard against a
future contributor adding one "for convenience".

### Overfitting

*What is done.* Shallow trees and large leaves by default; ridge and elastic-net
rather than unregularised OLS; scikit-learn's internal `early_stopping` switched
**off** because its validation split is random, which on time-series data would
validate on shuffled future observations. Model selection is the walk-forward
runner's job, and it splits chronologically.

Every fold gets a fresh model via `model.clone()`. Reusing a fitted object would
let state from fold 3 influence fold 4.

Preprocessing lives **inside** the model pipeline, so the scaler is fitted on
training rows only. Scaling before splitting is a leak that is nearly invisible
in the metrics because it is small but systematic.

### Multiple testing and data snooping

*The problem.* Test 20 strategy variants at the 5% level and you expect one
"significant" result when none work. Backtesting is worse: every configuration
choice multiplies the effective trial count, and most are never reported.

*What is done.* Three corrections of increasing sophistication, in
`quantlab.research.multiple_testing`:

- **Bonferroni** — always valid, very conservative when tests are correlated.
- **Benjamini-Hochberg** — controls the false discovery rate; the right choice
  for research screening.
- **Deflated Sharpe ratio** (Bailey and López de Prado, 2014) — given that `N`
  configurations were tried, and given this strategy's skewness, kurtosis and
  track-record length, what is the probability its true Sharpe is above zero?

The API's `/backtests/{id}/performance` endpoint uses the Sharpe ratios of
**every backtest in the database** as the trial distribution. That is a lower
bound — configurations abandoned before being persisted do not appear — and the
docstring says so.

### Survivorship bias

*The problem.* Backtesting today's index members over history measures the
performance of companies that survived.

*What is done, partly.* The schema has `universes` and `universe_members` with
`valid_from` / `valid_to`, and `repo.universe_symbols_as_of` resolves membership
at a date. `assets` records `listed_on`, `delisted_on` and `status`.

*What cannot be done with free data.* Delisted tickers are generally not
downloadable from Stooq. The machinery exists and is tested, but the demo
universe is ETFs that still trade. **Any equity result from this platform on free
data carries survivorship bias.** ETFs are used partly for this reason: an index
fund's history does not have the same survivorship problem as a stock's.

### Timestamp misalignment

*What is done.* Everything is timezone-aware UTC internally. Naive datetimes are
**rejected** at the boundary rather than coerced, because silent coercion of a
naive local timestamp is itself a source of misalignment. A `UTCDateTime`
column type restores awareness on read, since SQLite has no aware type and unit
tests would otherwise exercise naive timestamps while production used aware ones.

Session-close semantics live in one place, `SessionSpec`, which uses `zoneinfo`
so daylight saving is handled — a January US bar becomes available at 21:00 UTC,
a July one at 20:00.

## Evaluation

### Forecast accuracy

The headline metric is the **Campbell-Thompson out-of-sample R²**:

```
R²_OS = 1 - MSE(model) / MSE(benchmark)
```

with the benchmark a zero forecast. Ordinary R² compares against the mean of the
*test* sample, a quantity nobody knew in advance.

Calibration: in that literature, values above ~0.005 on monthly equity returns
are considered economically meaningful. **Anything above 0.05 on daily data
should be treated as a bug until proven otherwise**, and the first thing to check
is look-ahead bias.

The **information coefficient** is reported alongside, in Pearson and Spearman
form. A model can have a useless R² and still be tradable if it ranks correctly,
and financial returns are heavy-tailed enough that a few observations can
dominate a Pearson correlation.

### Statistical significance

The **Diebold-Mariano** test compares two forecasts' losses directly. Comparing
two RMSE numbers without a test says nothing about whether the gap would survive
on new data.

The Harvey-Leybourne-Newbold small-sample correction is applied and the statistic
compared against a *t* distribution rather than a normal, because the asymptotic
version over-rejects badly at the sample sizes available here. For an `h`-step
forecast the long-run variance uses `h-1` Newey-West lags.

Regressions use HAC standard errors by default. Textbook errors on overlapping
returns inflate *t*-statistics by roughly `sqrt(h)`, which is how a great many
"significant" predictors in the literature were found.

### Economic significance

Forecasts are traded through the backtester at three cost levels:

| Scenario | Commission | Slippage | Purpose |
|---|---|---|---|
| `frictionless` | none | none | A control, not a tradable scenario |
| `realistic` | $0.005/share, $1 min | 5 bps | Roughly US retail |
| `pessimistic` | $0.01/share | 5 bps + sqrt impact | Stress case |

A model can be statistically better and economically worthless. The gap between
the two is usually where the interesting finding is.

### Confidence intervals

Sharpe ratios are point estimates from one sample. The **stationary block
bootstrap** (Politis and Romano, 1994) resamples geometric-length blocks, which
preserves autocorrelation and volatility clustering — unlike an iid bootstrap,
which destroys both. Typical result for three years of daily data: the interval
is roughly ±1 around the point estimate, which is a useful corrective to a
reported Sharpe of 1.2.

## Limitations that cannot be fixed with the available data

These are stated plainly because a portfolio project that hides them is worse
than one that has them.

1. **Price returns, not total returns.** Stooq returns a single close series,
   adjusted for splits. We do not assume dividend adjustment, so `adj_close =
   close` and every return is a price return. For dividend-paying assets this
   understates total return by roughly the dividend yield per year. Strategies
   and benchmarks are measured on the same basis, so relative comparisons stay
   valid; absolute levels do not.

2. **Revised macro data.** The free FRED CSV endpoint returns the series *as it
   stands today*, not as first published. A model trained on revised GDP is using
   numbers nobody had at the time. The publication-lag approximation fixes the
   *timing* but not the *values*. ALFRED exposes real-time vintages and is the
   correct upgrade; it is listed as future work rather than silently ignored.

3. **Survivorship.** As above.

4. **No intraday data.** Everything is daily. Execution is modelled at the next
   open or close; the fill an execution algorithm would actually achieve is not
   observable here.

5. **No borrow costs, margin interest, or interest on cash.** Long-short
   strategies are flattered; cash-heavy strategies are penalised.

6. **One market regime per sample.** A backtest from 2012 to 2023 covers one
   sustained equity bull market and two short crashes. Conclusions drawn from it
   are conclusions about that regime.

7. **The impact coefficient is a convention, not an estimate.** Calibrating it
   needs execution data this project does not have.

## Reproducibility

A result is reproducible when four values match:

| Hash | Covers |
|---|---|
| Feature-set spec hash | Every feature's function, lookback and parameters |
| Model config hash | Model type, hyperparameters, splitter configuration |
| Data fingerprint | Shape, columns and values of the training matrix |
| Seed | All stochastic components |

All four are written into `model_runs` and into a JSON manifest under
`data/artifacts/runs/`. Two runs agreeing on all four should produce the same
numbers, and the property is checkable by anyone with the repository — no
tracking server required. See
[`decisions/ADR-005-experiment-tracking.md`](decisions/ADR-005-experiment-tracking.md).

## Validating the validator

The strongest evidence that the pipeline is neither leaking nor broken is a
two-sided test on data whose answer is known
(`tests/integration/test_research_loop.py`):

- On a series with an **injected AR(1) component**, the pipeline must find the
  signal: information coefficient above 0.10, out-of-sample R² above 0.01,
  directional accuracy above 53%.
- On a **pure random walk**, it must find nothing: |IC| below 0.06, R² not
  positive, directional accuracy between 46% and 54%.

A system that reports signal on a random walk is leaking. One that finds nothing
in the AR(1) series is broken. Passing both directions is what makes a negative
result on real data believable.

## References

- Bailey, D. and López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
  *Journal of Risk*.
- Bailey, D. and López de Prado, M. (2014). The Deflated Sharpe Ratio.
  *Journal of Portfolio Management*.
- Campbell, J. and Thompson, S. (2008). Predicting Excess Stock Returns Out of
  Sample. *Review of Financial Studies*.
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal Versus Naive
  Diversification. *Review of Financial Studies*.
- Diebold, F. and Mariano, R. (1995). Comparing Predictive Accuracy.
  *Journal of Business and Economic Statistics*.
- Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the Equality of
  Prediction Mean Squared Errors. *International Journal of Forecasting*.
- Ledoit, O. and Wolf, M. (2004). A Well-Conditioned Estimator for
  Large-Dimensional Covariance Matrices. *Journal of Multivariate Analysis*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Politis, D. and Romano, J. (1994). The Stationary Bootstrap.
  *Journal of the American Statistical Association*.

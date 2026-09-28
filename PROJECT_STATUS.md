# Project status

Implementation and verification notes for readers of this repository.

**Last updated:** 2026-09-28.

| | |
|---|---|
| Source | Installable `quantlab` package under `src/quantlab/` |
| Tests | 558 passed, 1 skipped on Python 3.12 (2026-09-28); coverage not measured |
| Docs | README, topic guides, architecture decisions, and generated example reports |
| Quality gates | Ruff lint and format, plus mypy, pass locally on Python 3.12 |
| Database | 19 ORM tables and 2 Alembic migrations; live PostgreSQL check not run locally |

Mypy targets Python 3.12, matching its CI job. The test matrix also includes
Python 3.10, the declared runtime floor, but that job has not been observed on
GitHub for the current working tree.

---

## 1. Fully implemented

### Data engineering
- **Provider framework** with a `fetch` / `parse` split: I/O separate from a pure
  transformation, so every parser is tested against checked-in bytes with no
  network.
- **Three providers**: Stooq (daily bars, no key), FRED (macro CSV, no key), and
  a seeded synthetic generator that implements the same interface.
- **Async HTTP** with a token-bucket rate limiter, retries with full-jitter
  exponential backoff, `Retry-After` handling, and a retry policy that never
  retries a 404.
- **Raw layer**: content-addressed, immutable, with a manifest per payload.
  Written before any parsing.
- **Validation**: a hard schema gate that stops the pipeline, plus nine soft
  quality checks persisted to `data_quality_checks`.
- **Idempotent incremental loading**: upserts on natural keys, watermark from the
  database with an overlap window for provider revisions.

### Database
- 19 tables with primary keys, foreign keys, indexes on every hot path, and
  CHECK constraints encoding the domain rules.
- Alembic migrations; CI runs `alembic check` against migrated PostgreSQL to
  catch schema differences supported by Alembic autogenerate.
- `UTCDateTime` column type so SQLite (unit tests) and PostgreSQL (production)
  both return timezone-aware datetimes.
- Point-in-time universe membership tables for survivorship handling.

### Backtesting engine
- Event-driven loop with explicit orders, fills, positions and cash.
- Execution at next open or next close; same-bar execution is unexpressible.
- Commission (per-share, percent), slippage (fixed spread, square-root impact),
  participation caps, position sizing (target weight, volatility target).
- Long and short, benchmark tracking, full metric suite.
- Accounting identity verified after every bar.

### Temporal-leakage protection
- `available_at` on every fact row, enforced by database constraints.
- `PointInTimeFrame` / `PointInTimeUniverse` structural guards.
- Behavioural leakage harness catching eight deliberately-broken feature
  functions.
- Purged, embargoed walk-forward splitters with no shuffle option anywhere.

### Machine learning
- Common `Model` interface with `clone()` so each fold gets a fresh estimator.
- Ladder: `zero`, `historical_mean`, `last_value`, `ridge`, `lasso`,
  `elastic_net`, `logistic`, `random_forest`, `gradient_boosting`, optional
  `xgboost`.
- Walk-forward runner producing predictions stamped with honest availability.
- Experiment tracking via four content hashes, JSON manifests and `model_runs`.

### Statistical research
- Descriptive statistics, correlation, drawdown tables.
- ADF and KPSS with a four-way verdict; ACF/PACF; Ljung-Box; ARCH-effect check.
- OLS with Newey-West HAC standard errors, horizon-aware.
- Diebold-Mariano with the Harvey-Leybourne-Newbold correction. Panel comparisons
  average paired loss differences by date and apply HAC across dates.
- Stationary block bootstrap.
- Bonferroni, Benjamini-Hochberg, probabilistic and deflated Sharpe ratio,
  minimum track record length.

### Risk and portfolio
- VaR in three flavours (historical, parametric, Cornish-Fisher), CVaR, rolling
  VaR with breach-rate backtesting.
- Ledoit-Wolf covariance shrinkage.
- Equal weight, inverse volatility, minimum variance, risk parity, maximum
  Sharpe, efficient frontier, risk contributions.

### Orchestration, serving, infrastructure
- Five Airflow DAGs containing only schedules and dependency graphs.
- CLI with eleven subcommands, each mapping to one pipeline function.
- FastAPI with 21 endpoints, Pydantic models, typed error handling, and `as_of`
  point-in-time queries.
- Streamlit dashboard reading through the API.
- Multi-stage Docker images, compose with three profiles, GitHub Actions with six
  jobs.

### Research study
- `quantlab.research_run` runs the four-stage study and writes a Markdown report
  and a JSON result file; a generated example is committed at
  `docs/example-research-report.md`. The verdict is generated from the numbers and has
  branches for every outcome including a negative one.

---

## 2. Partially implemented

| Area | What works | What is missing |
|---|---|---|
| **Survivorship handling** | Schema, point-in-time membership queries, tests | No delisted-symbol data to populate it; free sources do not provide it |
| **Macro point-in-time** | Conservative publication lags, availability-based joins | Real vintages need ALFRED; current values are revised ones |
| **Classification models** | Interface, logistic, RF and GB classifiers, AUC/log-loss/Brier | The default ladder and the research study are regression-only |
| **Airflow** | DAGs, images, compose profile, structural tests | Never run against a live scheduler in this session; the DagBag test skips without Airflow installed |
| **Dashboard** | Market data, backtests, models, risk, monthly returns | No feature explorer, no live optimiser page |
| **Docker** | Images and compose written and `docker compose config` validated | The build was never executed here; no Docker daemon in the build environment |

---

## 3. What remains

**Further validation and research work:**

1. **Complete the real-data study.** A close-only S&P 500 forecast pilot using
   FRED data ran on 2026-09-27; see `docs/real-market-pilot.md`. It did not
   test trading returns or costs. Stooq returned a browser challenge and Yahoo
   a rate-limit response, so the project's ETF OHLCV ingestion and full
   `make research` workflow remain unverified on market data.
2. **Build and run the Docker stack once.** The configuration is written but
   unexecuted. Expect small fixes.
3. **Start the Airflow scheduler once** and confirm the DagBag loads and one DAG
   runs end to end.

**Worth adding next, in order of value:**

4. Cross-sectional (rank-based) features and labels. Time-series prediction of a
   single asset's return is the hardest version of the problem; relative
   ranking within a universe is where most published edge lives.
5. ALFRED vintages, removing the largest remaining approximation.
6. Regime-conditional evaluation: report metrics separately for high- and
   low-volatility periods.
7. A combinatorially purged cross-validation splitter for more robust estimates.
8. Model persistence, if a fitted model ever needs serving.

---

## 4. Known issues and rough edges

| Issue | Severity | Detail |
|---|---|---|
| Docker images never built | Medium | Written and validated with `docker compose config`, but not executed |
| Airflow never run live | Medium | Structural tests pass; the scheduler was never started |
| Cash earns no interest | Low | Penalises strategies that sit in cash; documented in `docs/backtesting.md` |
| No borrow costs | Low | Flatters short strategies |
| `mypy` needs a cache directory outside a network mount | Cosmetic | `MYPY_CACHE_DIR=/tmp/mypy_cache mypy` if SQLite reports a disk I/O error |
| Coverage run is slow | Cosmetic | ~5 minutes with `--cov`; ~95 seconds without |
| Prediction strategy is threshold-based | Low | A rank-based version would use forecasts better; the hook exists |

Two design errors were found by tests during development and fixed. They are
recorded below because they show what the checks can catch.

---

## 5. Architectural decisions, and why

Six are written up in `docs/decisions/`. The condensed version:

| Decision | Why | Cost accepted |
|---|---|---|
| `available_at` on every row | Turns look-ahead from a code-review question into a query predicate | One extra column everywhere; availability is derived, not observed |
| Event-driven backtester | A vectorised backtest's correctness rests on one `shift` nobody can verify by reading | ~700 lines where 3 would do; needed a performance fix |
| Free providers + synthetic generator | Clone and run, no keys, deterministic offline CI | Price returns not total returns; no delisted data |
| PostgreSQL with CHECK constraints | Invariants enforced where no application bug can bypass them | A service to run; SQLite needed a custom datetime type |
| Content hashes, not MLflow | Reproducibility that does not depend on a server | No experiment-comparison UI |
| Installable `src/` package | One import path in pytest, Airflow, Docker and notebooks | One more directory level |
| Weights, not share counts, from strategies | Scale-free; sizing and alpha vary independently | An indirection to explain |
| Long-format feature storage | Adding a feature is an insert, not a migration | Larger table, pivot on read |
| Airflow behind a compose profile | A reviewer should not wait for a scheduler to boot | Two commands instead of one |

### Two errors the tests caught

These are worth being able to tell, because "what went wrong and how did you
find it" is a question you will be asked.

**1. The leakage harness had a false positive.** The first version corrupted
future data with values 1000× the series' scale, and flagged `rolling().skew()`
as leaky. Investigating showed pandas mean-centres the *entire* array before its
sliding skew/kurt/var computation, for numerical stability. Changing the tail by
three orders of magnitude changes that constant and therefore the rounding of
head values. The head was not reading the future; the library was re-centring.
Fixed by corrupting at the series' own scale, and the artefact is reproduced in a
test so the reasoning is verified rather than asserted in a comment.

**2. A CHECK constraint encoded a false assumption.** `model_runs` originally had
`CHECK (test_start >= train_end)`. Correct for a single holdout split, wrong for
walk-forward, where fold 3 trains on data fold 1 already tested on, so the
aggregate spans overlap. The first multi-fold run failed to persist. Migration
0002 drops it; the per-fold guarantee is enforced where it is actually true, in
`assert_split_ordering` and in `predictions`' `CHECK (target_ts > ts)`.

There was also one performance fix worth mentioning: the engine's point-in-time
view was `O(n²)`, making a fifteen-year backtest take minutes. Fixed with prefix
slicing and binary search — 19.1s to 1.75s, byte-identical results. It mattered
for correctness, not comfort: when the safe path is slow, people stop using it.

---

## 6. Commands

### Run everything with Docker

```bash
docker compose --profile demo up -d --build     # Postgres, migrations, demo data, API, dashboard
docker compose --profile airflow up -d --build  # add Airflow at :8080 (airflow/airflow)
docker compose logs -f
docker compose --profile demo --profile airflow down -v
```

| Service | URL |
|---|---|
| API + OpenAPI docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Airflow | http://localhost:8080 |
| PostgreSQL | `localhost:5432`, `quantlab`/`quantlab` |

### Run everything locally

```bash
python -m venv .venv && source .venv/bin/activate
make install                 # pip install -e ".[api,dashboard,dev]"
cp .env.example .env

make migrate                 # alembic upgrade head
make seed                    # deterministic synthetic dataset, no network
make api                     # http://localhost:8000
make dashboard               # http://localhost:8501
```

### The pipeline, step by step

```bash
quantlab status                    # connectivity and row counts
quantlab config                    # resolved settings, secrets redacted

quantlab ingest-prices             # demo universe from Stooq
quantlab ingest-macro              # FRED series
quantlab returns                   # trailing and forward returns
quantlab validate                  # warehouse checks
quantlab features                  # build and store the feature set
quantlab train                     # walk-forward the model ladder
quantlab backtest                  # classical strategies

make research                      # the full study -> data/artifacts/RESEARCH_REPORT.md
make research-offline              # same, synthetic data, no database
```

### Tests

```bash
make test                          # full suite; verify the current result
make test-fast                     # unit + leakage only, ~15s
make test-leakage                  # the look-ahead suite alone
make coverage                      # with an HTML report (slow, ~5 min)

pytest -m unit                     # current unit tests
pytest -m integration              # current integration tests
pytest -m leakage                  #  66 tests
pytest tests/leakage/test_temporal_execution.py -v    # the central claim
pytest -k "leakage_harness" -v                        # the eight broken features
```

### Quality gates

```bash
make check                         # lint + typecheck + test, what CI runs
make lint                          # ruff check + format --check
make typecheck                     # mypy
make format                        # apply fixes

# If mypy reports a disk I/O error (network-mounted working directory):
MYPY_CACHE_DIR=/tmp/mypy_cache mypy
```

### Database

```bash
make migrate                       # alembic upgrade head
make migration m="add x"           # autogenerate
make downgrade                     # roll back one
```

---

## 7. Concepts you need to be able to defend

Ordered by how likely they are to come up.

### Tier 1 — you will be asked about these

**Look-ahead bias and the availability model.** Be able to explain the `ts` vs
`available_at` distinction in one sentence with a concrete example (January CPI),
and name the four layers that enforce it. Know that the January CPI example is
the one that makes it click.

**Why random train/test splitting is wrong for time series.** Because it trains
on 2023 and tests on 2019, which is interpolation, not forecasting. Be able to
say what purging and embargo are and why an `h`-day label creates the problem.

**Out-of-sample R² against a benchmark.** `1 - MSE(model)/MSE(benchmark)`, why
the benchmark must be something knowable in advance, and why 0.05 on daily data
should be assumed to be a bug.

**Transaction costs as the mechanism, not the footnote.** Point at the turnover
column in the current offline results: buy-and-hold turns over about 0.46 times
a year; ridge turns over about 98.6 times and has a lower Sharpe after costs.

**Why your result is negative and why that is fine.** Have the sentence ready:
*"In the synthetic example, three of four ML models had negative out-of-sample
R² and significantly higher loss than a zero forecast. This does not tell us
how they perform on market data."*

### Tier 2 — likely follow-ups

**Diebold-Mariano.** Tests equal predictive accuracy between two forecasts. Why
you need it: comparing two RMSE numbers says nothing about whether the gap
survives on new data. Why the small-sample correction: the asymptotic version
over-rejects at these sample sizes.

**Deflated Sharpe ratio.** Given `N` configurations tried, the expected maximum
Sharpe of `N` random strategies is not zero. Deflation asks whether the observed
Sharpe clears that bar. In the current synthetic example, the probabilistic
Sharpe is 99.995% and the deflated Sharpe is 99.54%. Neither is market evidence.

**Benjamini-Hochberg vs Bonferroni.** FDR vs family-wise error rate; BH is less
conservative and right for screening.

**The stationary block bootstrap.** Why an iid bootstrap is wrong for returns
(destroys autocorrelation and volatility clustering) and why blocks fix it.

**ADF vs KPSS.** Opposite nulls. Why running only one conflates "evidence of
stationarity" with "failure to reject non-stationarity".

**Newey-West standard errors.** Overlapping returns are mechanically
autocorrelated; textbook standard errors inflate t-statistics by roughly
`sqrt(h)`.

**Ledoit-Wolf shrinkage and why minimum-variance beats maximum-Sharpe out of
sample.** Mean-variance optimisation is error-maximising: it puts the largest
weights on the assets whose returns are most overestimated.

### Tier 3 — engineering questions

- Why `src/` layout (one import path everywhere).
- Why event-driven rather than vectorised (auditability).
- Why NUMERIC for money and DOUBLE PRECISION for statistics.
- Why enums are VARCHAR + CHECK rather than native PostgreSQL types.
- What makes a pipeline idempotent here (content-addressed raw layer, upserts on
  natural keys, watermark from the database not the scheduler).
- Why the scaler lives inside the model pipeline.
- Why `clone()` per fold.

### Reading list

- López de Prado, *Advances in Financial Machine Learning* — chapters 3 (labels),
  7 (cross-validation), 8 (feature importance). Chapter 7 is the source of
  purging and embargo.
- Bailey and López de Prado (2014), *The Deflated Sharpe Ratio* — short, and the
  source of the correction in `research/multiple_testing.py`.
- Campbell and Thompson (2008) — the out-of-sample R² and why small values matter.
- Diebold and Mariano (1995) — the test, and Harvey et al. (1997) for the
  correction.

---

## 8. Files to study first

In this order. The whole path is about eight hours of reading.

### The core idea (start here, ~1 hour)

1. **`src/quantlab/timeutils.py`** — the availability model. Read the module
   docstring; everything else follows from it.
2. **`src/quantlab/pit.py`** — how the future is made structurally unreachable.
3. **`tests/leakage/test_temporal_execution.py`** — the tests that prove the
   central claim, especially `test_the_overnight_gap_is_not_captured` and its
   complement.

### The backtester (~2 hours)

4. **`src/quantlab/backtesting/engine.py`** — read `BacktestEngine.run` line by
   line. The three-phase bar loop is the heart of the project.
5. **`src/quantlab/backtesting/execution.py`** — small, and contains the single
   assertion the temporal claim rests on.
6. **`src/quantlab/backtesting/portfolio.py`** — the accounting identity.
7. **`tests/unit/test_backtest_engine.py`** — the closed-form arithmetic tests
   have their derivations in the docstrings.

### Leakage protection (~1.5 hours)

8. **`src/quantlab/features/guards.py`** — the harness, including the pandas
   re-centring story in the module docstring.
9. **`src/quantlab/features/transforms.py`** — the causal primitives and the list
   of banned operations.
10. **`tests/leakage/test_leakage_harness.py`** — the eight broken functions.
11. **`src/quantlab/models/splitters.py`** — purging and embargo.

### Data engineering (~1.5 hours)

12. **`src/quantlab/db/models.py`** — the schema, with the reasoning in the
    docstrings.
13. **`src/quantlab/ingestion/base.py`** — the fetch/parse split.
14. **`src/quantlab/ingestion/pipeline.py`** — the three idempotency mechanisms.
15. **`src/quantlab/ingestion/http.py`** — retry policy and why full jitter.

### The research (~2 hours)

16. **`src/quantlab/research_run.py`** — the four-stage study and, importantly,
    `_build_verdict`, which is where the honesty is enforced in code.
17. **`src/quantlab/research/multiple_testing.py`** — the deflated Sharpe ratio.
18. **`src/quantlab/research/hypothesis.py`** — Diebold-Mariano and the bootstrap.
19. **`tests/integration/test_research_loop.py`** — the two-sided validation.
20. **`docs/methodology.md`** — read last, once the code makes sense.

### If you only have thirty minutes

`timeutils.py` docstring → `execution.py` → `test_temporal_execution.py` →
the Results section of the README.

---

## 9. Recommended next steps

**This week**
1. Run `quantlab ingest-prices` on a normal network, then `make research`.
   Replace the README Results section with real numbers, keeping the honest
   framing whichever way it comes out.
2. `docker compose --profile demo up --build` once; fix whatever surfaces.
3. Start Airflow once and confirm one DAG runs end to end.

**This month**
4. Add cross-sectional features and a ranking label. This is the single highest
   expected-value addition: relative ranking within a universe is where most
   published edge lives, and the platform already has everything else needed.
5. Add regime-conditional reporting.
6. Write a two-page summary of the research finding, suitable for an application
   portfolio, referencing the generated report.

**Before submitting the application**
7. Re-read `PROJECT_STATUS.md` section 7 and rehearse the Tier 1 answers out
   loud. Being able to explain *why* each choice was made matters more than the
   code volume.
8. Have one concrete story ready about something that went wrong and how you
   found it. The pandas re-centring false positive is the best one: it shows you
   investigated an anomaly instead of loosening a threshold.

"""The research study.

Answers the question the platform was built for:

    Do machine-learning models provide statistically and economically meaningful
    improvements over simple statistical baselines for financial time-series
    prediction, after realistic transaction costs?

The study is deliberately structured so that a negative answer is as easy to
report as a positive one.  There is no step that selects the best model and
reports only that; every model in the ladder appears in the output with the same
metrics, and the multiple-testing correction is applied to the whole set.

Four stages
-----------
1.  **Statistical accuracy.**  Walk-forward evaluation of every model on
    identical folds.  Reported as out-of-sample R-squared against a zero
    forecast, plus information coefficient and directional accuracy.

2.  **Statistical significance.**  A Diebold-Mariano test of each model's
    forecast errors against the zero baseline, with Benjamini-Hochberg
    correction across models.  Comparing two RMSE numbers without a test says
    nothing about whether the gap would survive on new data.

3.  **Economic significance.**  Each model's forecasts are traded through the
    backtester at three cost levels, from frictionless to pessimistic.  A model
    can be statistically better and economically worthless, and the gap between
    the two is usually where the interesting finding is.

4.  **Data-snooping correction.**  The deflated Sharpe ratio across every
    strategy evaluated, so the headline number accounts for how many were tried.

Nothing here manufactures a result.  If the ML models lose, the report says so.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtesting.costs import (
    ZERO_COST,
    CostModel,
    FixedBpsSlippage,
    PerShareCommission,
    SquareRootImpactSlippage,
)
from quantlab.backtesting.engine import BacktestEngine, EngineConfig
from quantlab.backtesting.execution import ExecutionModel
from quantlab.backtesting.market import MarketData
from quantlab.backtesting.results import build_result
from quantlab.backtesting.sizing import TargetWeightSizer
from quantlab.backtesting.strategy import (
    BuyAndHold,
    MovingAverageCrossover,
    PredictionStrategy,
    TimeSeriesMomentum,
)
from quantlab.config import get_settings
from quantlab.features.library import default_feature_set
from quantlab.features.pipeline import build_training_frame
from quantlab.logging import get_logger
from quantlab.models.registry import DEFAULT_REGRESSION_LADDER, get_model
from quantlab.models.splitters import ExpandingWindowSplitter, SplitterConfig
from quantlab.models.walkforward import WalkForwardResult, WalkForwardRunner
from quantlab.research.hypothesis import (
    diebold_mariano_loss_differential,
    sharpe_statistic,
    stationary_bootstrap_ci,
)
from quantlab.research.multiple_testing import adjust_pvalues, analyse_sharpe
from quantlab.research.stationarity import classify_stationarity
from quantlab.timeutils import UTC

log = get_logger(__name__)

#: Three cost regimes.  ``frictionless`` is not a scenario anyone can trade; it
#: is the control that isolates how much of the result costs destroy.
COST_SCENARIOS: dict[str, CostModel] = {
    "frictionless": ZERO_COST,
    "realistic": CostModel(
        commission=PerShareCommission(rate_per_share=0.005, minimum=1.0),
        slippage=FixedBpsSlippage(5.0),
    ),
    "pessimistic": CostModel(
        commission=PerShareCommission(rate_per_share=0.01, minimum=1.0),
        slippage=SquareRootImpactSlippage(half_spread_bps=5.0, impact_coefficient=1.0),
    ),
}

OFFLINE_SYMBOLS: tuple[str, ...] = ("SYN_BROAD", "SYN_GROWTH", "SYN_DEFENSIVE", "SYN_TREND")

#: Models that are *baselines*, not machine learning.  The research question asks
#: whether ML beats these, so the verdict must not count one baseline
#: outperforming another as evidence for ML.  Keeping the set explicit stops that
#: category error, which is easy to make when every model is just a table row.
BASELINE_MODELS: frozenset[str] = frozenset(
    {"zero", "historical_mean", "last_value", "majority_class"}
)


@dataclass(slots=True)
class StudyResult:
    generated_at: dt.datetime
    data_source: str
    symbols: list[str]
    horizon: int
    n_observations: int
    n_features: int
    feature_set_hash: str
    sample_start: str
    sample_end: str
    evaluation_start: str
    evaluation_end: str
    forecast_table: pd.DataFrame
    significance_table: pd.DataFrame
    economic_table: pd.DataFrame
    stationarity: dict[str, Any]
    snooping: dict[str, Any]
    verdict: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "data_source": self.data_source,
            "symbols": self.symbols,
            "horizon_days": self.horizon,
            "n_observations": self.n_observations,
            "n_features": self.n_features,
            "feature_set_hash": self.feature_set_hash,
            "sample": {"start": self.sample_start, "end": self.sample_end},
            "evaluation_window": {
                "start": self.evaluation_start,
                "end": self.evaluation_end,
                "note": "every strategy, references included, is scored on this window",
            },
            "forecast_accuracy": self.forecast_table.reset_index().to_dict("records"),
            "statistical_significance": self.significance_table.to_dict("records"),
            "significance_method": (
                "Mean paired squared-loss difference across available assets on each date; "
                "Diebold-Mariano statistic with HAC standard errors across dates and "
                "Benjamini-Hochberg adjustment across models. n counts distinct dates."
            ),
            "economic_significance": self.economic_table.to_dict("records"),
            "stationarity": self.stationarity,
            "data_snooping": self.snooping,
            "verdict": self.verdict,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_price_panel(
    symbols: list[str] | None, use_database: bool, *, universe: str | None = None
) -> tuple[pd.DataFrame, str, list[str]]:
    """Prices from the warehouse, or from the synthetic generator when offline.

    ``universe`` selects a point-in-time membership list.  Without it the
    database branch falls back to every asset currently in ``assets``, which is
    survivorship-biased -- see :func:`quantlab.pipelines.resolve_universe`, which
    owns that decision and logs the fallback.
    """
    if use_database:
        from quantlab.db import repository as repo
        from quantlab.db.session import session_scope
        from quantlab.pipelines import resolve_universe

        with session_scope() as session:
            resolved = resolve_universe(session, symbols, universe=universe, context="study")
            prices = repo.load_prices(session, resolved)
        if prices.empty:
            raise RuntimeError(
                "no prices in the database; run `quantlab ingest-prices` or "
                "`quantlab research --offline`"
            )
        return prices, "database", sorted(prices["symbol"].unique())

    from quantlab.ingestion.providers.synthetic import SyntheticSpec, generate_price_frame

    resolved = symbols or list(OFFLINE_SYMBOLS)
    specs = {
        "SYN_BROAD": (0.07, 0.16, 0.0),
        "SYN_GROWTH": (0.09, 0.24, 0.0),
        "SYN_DEFENSIVE": (0.04, 0.09, 0.0),
        "SYN_TREND": (0.06, 0.18, 0.20),
    }
    frames = []
    for index, symbol in enumerate(resolved):
        drift, volatility, rho = specs.get(symbol, (0.06, 0.18, 0.0))
        frames.append(
            generate_price_frame(
                symbol,
                SyntheticSpec(
                    start=dt.date(2012, 1, 2),
                    end=dt.date(2023, 12, 29),
                    annual_drift=drift,
                    annual_volatility=volatility,
                    autocorrelation=rho,
                ),
                seed=index,
            )
        )
    return pd.concat(frames, ignore_index=True), "synthetic", resolved


def _daily_panel_loss_difference(joined: pd.DataFrame) -> pd.Series:
    """Cross-sectional mean of paired candidate minus baseline squared loss."""
    if not np.allclose(joined["y_true"], joined["y_true_base"], rtol=0, atol=1e-12):
        raise ValueError("candidate and baseline targets differ")
    return (
        (
            (joined["y_true"] - joined["y_pred"]) ** 2
            - (joined["y_true"] - joined["y_pred_base"]) ** 2
        )
        .groupby(level="ts")
        .mean()
        .sort_index()
    )


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------
def run_research_study(
    symbols: list[str] | None = None,
    *,
    horizon: int = 1,
    use_database: bool = True,
    n_splits: int = 5,
    test_size: int = 252,
    min_train_size: int = 756,
    output_dir: Path | None = None,
    universe: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    settings.ensure_directories()
    destination = Path(output_dir or settings.artifacts_dir)
    destination.mkdir(parents=True, exist_ok=True)

    prices, source, resolved = load_price_panel(symbols, use_database, universe=universe)
    feature_set = default_feature_set(include_macro=False)
    X, y, meta = build_training_frame(prices, feature_set, horizon=horizon)
    log.info("study.data_ready", source=source, rows=len(X), symbols=len(resolved))

    splitter = ExpandingWindowSplitter(
        SplitterConfig(
            n_splits=n_splits,
            test_size=test_size,
            min_train_size=min_train_size,
            embargo=pd.Timedelta(days=max(2, horizon + 1)),
        )
    )
    runner = WalkForwardRunner(splitter)

    # --- stage 1: forecast accuracy --------------------------------------
    results: dict[str, WalkForwardResult] = {}
    rows: list[dict[str, Any]] = []
    for name, params in DEFAULT_REGRESSION_LADDER:
        model = get_model(name, **params)
        try:
            result = runner.run(model, X, y, meta, seed=settings.random_seed)
        except Exception as exc:
            log.error("study.model_failed", model=name, error=str(exc))
            continue
        results[name] = result
        assert result.overall is not None
        rows.append({"model": name, **result.overall.to_dict()})

    if "zero" not in results:
        raise RuntimeError("the zero baseline failed to run; the comparison is meaningless")

    forecast_table = pd.DataFrame(rows).set_index("model")

    # --- stage 2: statistical significance -------------------------------
    baseline = results["zero"].predictions.set_index(["symbol", "ts"]).sort_index()
    significance_rows: list[dict[str, Any]] = []
    for name, result in results.items():
        if name == "zero":
            continue
        candidate = result.predictions.set_index(["symbol", "ts"]).sort_index()
        joined = candidate.join(
            baseline[["y_true", "y_pred"]], how="inner", rsuffix="_base"
        ).dropna(subset=["y_true", "y_pred", "y_true_base", "y_pred_base"])
        daily_loss_difference = _daily_panel_loss_difference(joined)
        if len(daily_loss_difference) < 30:
            continue
        outcome = diebold_mariano_loss_differential(daily_loss_difference, horizon=horizon)
        significance_rows.append(
            {
                "model": name,
                "n": int(len(daily_loss_difference)),
                "n_asset_date_pairs": int(len(joined)),
                "dm_statistic": outcome.statistic,
                "p_value": outcome.pvalue,
                "hac_lags": outcome.detail["hac_lags"],
                "favours": outcome.detail["favours"],
            }
        )

    significance_table = pd.DataFrame(significance_rows)
    if not significance_table.empty:
        adjusted = adjust_pvalues(
            dict(zip(significance_table["model"], significance_table["p_value"], strict=True)),
            method="benjamini_hochberg",
        )
        significance_table = significance_table.merge(
            adjusted[["adjusted_p_value"]].reset_index().rename(columns={"index": "model"}),
            on="model",
            how="left",
        )
        significance_table["beats_baseline_at_5pct"] = (
            significance_table["adjusted_p_value"] < 0.05
        ) & (significance_table["favours"] == "a")

    # --- stage 3: economic significance ----------------------------------
    #
    # Every strategy is evaluated over exactly the same window: the span for
    # which out-of-sample predictions exist.
    #
    # An earlier version ran the reference strategies over the whole panel while
    # the model strategies could only trade inside their test folds, because
    # that is the only period they have forecasts for.  On a twelve-year panel
    # with a one-year test span that gave buy-and-hold twelve years of
    # compounding and the models one, and counted the models' eleven flat years
    # in their annualised return and volatility.  The resulting table looked
    # like a decisive win for buy-and-hold and was measuring the window, not the
    # strategy.  Restricting every run to the common window is the only
    # comparison that means anything.
    prediction_times = pd.concat(
        [r.predictions["ts"] for r in results.values() if not r.predictions.empty]
    )
    oos_start = pd.Timestamp(prediction_times.min()).to_pydatetime()
    oos_end = pd.Timestamp(prediction_times.max()).to_pydatetime()
    log.info(
        "study.evaluation_window",
        start=str(oos_start.date()),
        end=str(oos_end.date()),
        note="all strategies, references included, are scored on this window only",
    )

    market = MarketData(prices)
    economic_rows: list[dict[str, Any]] = []
    strategy_returns: dict[str, np.ndarray] = {}

    for scenario, costs in COST_SCENARIOS.items():
        config = EngineConfig(
            initial_cash=1_000_000.0,
            execution=ExecutionModel(costs=costs, max_participation=0.1),
            sizer=TargetWeightSizer(allow_fractional=False, max_participation=0.05),
            rebalance="daily",
            rebalance_tolerance=0.05,
            benchmark_symbol=resolved[0],
            max_borrow_fraction=0.0,
        )
        engine = BacktestEngine(config)

        reference_strategies = {
            "buy_and_hold": BuyAndHold(resolved),
            "ma_crossover": MovingAverageCrossover(resolved, fast=20, slow=100),
            "ts_momentum": TimeSeriesMomentum(resolved, lookback=126, allow_short=False),
        }
        for label, strategy in reference_strategies.items():
            output = engine.run(strategy, market, start=oos_start, end=oos_end)
            metrics = build_result(output).metrics
            economic_rows.append(
                {
                    "scenario": scenario,
                    "strategy": label,
                    "kind": "reference",
                    **_economics(metrics),
                }
            )
            if scenario == "realistic":
                strategy_returns[label] = _returns_from(output)

        for name, result in results.items():
            if name == "zero":
                continue  # a constant zero forecast places no trades
            output = engine.run(
                PredictionStrategy(resolved, long_threshold=0.0, max_positions=len(resolved)),
                market,
                start=oos_start,
                end=oos_end,
                predictions=result.predictions,
            )
            metrics = build_result(output).metrics
            economic_rows.append(
                {
                    "scenario": scenario,
                    "strategy": name,
                    # A baseline forecast traded as a strategy is still a
                    # baseline.  Labelling it "model" is the same category error
                    # the verdict logic guards against.
                    "kind": "baseline" if name in BASELINE_MODELS else "model",
                    **_economics(metrics),
                }
            )
            if scenario == "realistic":
                strategy_returns[name] = _returns_from(output)

    economic_table = pd.DataFrame(economic_rows)

    # --- stage 4: data snooping ------------------------------------------
    realistic = economic_table[economic_table["scenario"] == "realistic"]
    sharpes = realistic["sharpe_ratio"].dropna().to_numpy(dtype="float64") / np.sqrt(252.0)
    best_label = (
        realistic.loc[realistic["sharpe_ratio"].idxmax(), "strategy"]
        if realistic["sharpe_ratio"].notna().any()
        else None
    )
    snooping: dict[str, Any] = {"n_strategies_evaluated": int(len(sharpes))}
    if best_label and best_label in strategy_returns and len(sharpes) > 1:
        best_returns = strategy_returns[best_label]
        inference = analyse_sharpe(best_returns, n_trials=len(sharpes), trial_sharpes=sharpes)
        snooping.update({"best_strategy": best_label, **inference.to_dict()})
        if len(best_returns) >= 60:
            snooping["sharpe_confidence_interval"] = stationary_bootstrap_ci(
                best_returns, sharpe_statistic(252), n_boot=500, seed=settings.random_seed
            )

    # --- diagnostics ------------------------------------------------------
    first_symbol = resolved[0]
    series = prices.loc[prices["symbol"] == first_symbol].sort_values("ts")["adj_close"]
    stationarity = {
        "symbol": first_symbol,
        "prices": classify_stationarity(series)["verdict"],
        "returns": classify_stationarity(series.pct_change(fill_method=None).dropna())["verdict"],
    }

    verdict, notes = _build_verdict(
        forecast_table, significance_table, economic_table, source, snooping
    )

    study = StudyResult(
        generated_at=dt.datetime.now(tz=UTC),
        data_source=source,
        symbols=resolved,
        horizon=horizon,
        n_observations=int(len(X)),
        n_features=int(X.shape[1]),
        feature_set_hash=feature_set.spec_hash(),
        sample_start=str(pd.to_datetime(meta["ts"]).min().date()),
        sample_end=str(pd.to_datetime(meta["ts"]).max().date()),
        evaluation_start=str(oos_start.date()),
        evaluation_end=str(oos_end.date()),
        forecast_table=forecast_table,
        significance_table=significance_table,
        economic_table=economic_table,
        stationarity=stationarity,
        snooping=snooping,
        verdict=verdict,
        notes=notes,
    )

    json_path = destination / "research_results.json"
    json_path.write_text(json.dumps(study.to_dict(), indent=2, default=str), encoding="utf-8")
    report_path = destination / "RESEARCH_REPORT.md"
    report_path.write_text(render_report(study), encoding="utf-8")
    log.info("study.complete", report=str(report_path), verdict=verdict[:80])

    return {
        "report": str(report_path),
        "results": str(json_path),
        "verdict": verdict,
        "n_models": len(results),
        "n_observations": int(len(X)),
        "data_source": source,
    }


def _economics(metrics: Any) -> dict[str, Any]:
    payload = metrics.to_dict()
    keys = (
        "total_return",
        "annualised_return",
        "annualised_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "annualised_turnover",
        "n_trades",
        "total_commission",
        "total_slippage",
    )
    return {k: payload.get(k) for k in keys}


def _returns_from(output: Any) -> np.ndarray:
    values = np.asarray([s.period_return for s in output.snapshots[1:]], dtype="float64")
    return values[np.isfinite(values)]


def _build_verdict(
    forecast: pd.DataFrame,
    significance: pd.DataFrame,
    economic: pd.DataFrame,
    source: str,
    snooping: dict[str, Any],
) -> tuple[str, list[str]]:
    """Write the conclusion from the numbers, in both directions."""
    notes: list[str] = []
    if source == "synthetic":
        notes.append(
            "This run used the synthetic generator, not market data. One series "
            "carries an injected autocorrelation, so a positive result here "
            "demonstrates that the pipeline detects signal that is known to "
            "exist; it says nothing about real markets."
        )

    ml_models = [m for m in forecast.index if m not in BASELINE_MODELS]
    baseline_models = [m for m in forecast.index if m in BASELINE_MODELS and m != "zero"]
    positive_r2 = [m for m in ml_models if (forecast.loc[m, "r2_oos_vs_zero"] or 0) > 0]

    beat_all = (
        significance.loc[significance["beats_baseline_at_5pct"], "model"].tolist()
        if "beats_baseline_at_5pct" in significance.columns
        else []
    )
    significant = [m for m in beat_all if m not in BASELINE_MODELS]
    significant_baselines = [m for m in beat_all if m in BASELINE_MODELS]

    if significant_baselines:
        notes.append(
            f"Baseline model(s) {significant_baselines} also beat the zero forecast "
            "significantly. That is a fact about the data, not evidence for machine "
            "learning, and it is excluded from the verdict."
        )
    if "adjusted_p_value" in significance.columns and not significance.empty:
        worse = significance.loc[
            (significance["favours"] == "b")
            & (significance["adjusted_p_value"] < 0.05)
            & (~significance["model"].isin(BASELINE_MODELS)),
            "model",
        ].tolist()
        if worse:
            notes.append(
                f"Model(s) {worse} were significantly *worse* than predicting zero. On a "
                "series with almost no conditional mean, a flexible model fits noise, and "
                "the extra variance shows up directly as forecast error."
            )
    if baseline_models:
        notes.append(f"Baselines evaluated alongside the models: {baseline_models}.")

    realistic = economic[economic["scenario"] == "realistic"]
    frictionless = economic[economic["scenario"] == "frictionless"]
    model_rows = realistic[realistic["kind"] == "model"]
    reference_best = realistic[realistic["kind"] == "reference"]["sharpe_ratio"].max()
    beats_reference = (
        model_rows.loc[model_rows["sharpe_ratio"] > reference_best, "strategy"].tolist()
        if pd.notna(reference_best)
        else []
    )

    if not frictionless.empty and not realistic.empty:
        cost_drag = (
            frictionless[frictionless["kind"] == "model"]["sharpe_ratio"].mean()
            - model_rows["sharpe_ratio"].mean()
        )
        if pd.notna(cost_drag):
            notes.append(
                f"Transaction costs reduce the average model Sharpe ratio by "
                f"{cost_drag:.2f} between the frictionless and realistic scenarios."
            )

    deflated = snooping.get("deflated_sharpe")
    if deflated is not None:
        notes.append(
            f"After deflating for the {snooping.get('n_trials')} strategy "
            f"configurations evaluated, the probability that the best strategy's "
            f"true Sharpe ratio is positive is {deflated:.1%}."
        )

    if not ml_models:
        verdict = "No machine-learning models were evaluated, so the question is unanswered."
    elif not positive_r2:
        verdict = (
            "No machine-learning model in the ladder achieved a positive out-of-sample "
            "R-squared against a zero forecast. On this sample the answer to the research "
            "question is no: the models do not improve on the naive baseline, before costs "
            "are even considered."
        )
    elif not significant:
        verdict = (
            f"{len(positive_r2)} model(s) achieved a positive out-of-sample R-squared, but "
            "no model's improvement over the zero baseline was statistically significant "
            "under a Diebold-Mariano test with Benjamini-Hochberg correction. The apparent "
            "edge is within sampling noise."
        )
    elif not beats_reference:
        verdict = (
            f"Model(s) {significant} beat the zero baseline statistically, but none produced "
            "a higher Sharpe ratio than the best simple reference strategy once realistic "
            "costs were applied. The improvement is statistically real and economically "
            "irrelevant, which is the most common honest outcome in this literature."
        )
    else:
        verdict = (
            f"Model(s) {beats_reference} beat both the zero baseline statistically and the "
            "best reference strategy economically under realistic costs. Treat this as "
            "provisional: read the deflated Sharpe ratio and the confidence interval below "
            "before concluding anything, and note that a single sample cannot establish an "
            "effect that is stable out of sample."
        )
    return verdict, notes


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(study: StudyResult) -> str:
    def table(frame: pd.DataFrame, columns: list[str] | None = None, index: bool = False) -> str:
        if frame.empty:
            return "_No results._\n"
        subset = frame[columns] if columns else frame
        return subset.round(6).to_markdown(index=index) + "\n"

    has_scenarios = not study.economic_table.empty and "scenario" in study.economic_table.columns
    forecast_columns = [
        "n",
        "rmse",
        "r2_oos_vs_zero",
        "information_coefficient",
        "rank_information_coefficient",
        "directional_accuracy",
    ]
    available = [c for c in forecast_columns if c in study.forecast_table.columns]

    lines = [
        "# Research report",
        "",
        "> **Research question.** Do machine-learning models provide statistically and",
        "> economically meaningful improvements over simple statistical baselines for",
        "> financial time-series prediction, after realistic transaction costs?",
        "",
        "This report is generated by `quantlab.research_run`. Every number in it comes",
        "from the run recorded below; nothing is hand-edited.",
        "",
        "## Verdict",
        "",
        study.verdict,
        "",
    ]
    if study.notes:
        lines.append("### Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in study.notes)
        lines.append("")

    lines += [
        "## Run",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Generated | {study.generated_at.isoformat()} |",
        f"| Data source | {study.data_source} |",
        f"| Symbols | {', '.join(study.symbols)} |",
        f"| Sample | {study.sample_start} to {study.sample_end} |",
        f"| Evaluation window | {study.evaluation_start} to {study.evaluation_end} |",
        f"| Observations | {study.n_observations:,} |",
        f"| Features | {study.n_features} |",
        f"| Feature set hash | `{study.feature_set_hash[:16]}` |",
        f"| Forecast horizon | {study.horizon} day(s) |",
        "",
        "## 1. Forecast accuracy",
        "",
        "Walk-forward evaluation on identical folds. `r2_oos_vs_zero` is the",
        "Campbell-Thompson out-of-sample R-squared against a zero forecast: values at or",
        "below zero mean the model is not beating a prediction of no change.",
        "",
        table(study.forecast_table[available], index=True),
        "",
        "## 2. Statistical significance",
        "",
        "For each date, average the paired squared-loss difference across available assets.",
        "The Diebold-Mariano statistic then uses dates as observations, with HAC standard",
        "errors across dates and Benjamini-Hochberg correction across models. `n` is",
        "the number of dates; `n_asset_date_pairs` is the number of paired forecasts.",
        "`favours = a` means the model has lower average loss than the zero baseline.",
        "",
        table(study.significance_table),
        "",
        "## 3. Economic significance",
        "",
        "Each forecast traded through the backtesting engine at three cost levels.",
        "`frictionless` is a control, not a tradable scenario.",
        "",
        f"**Every strategy below, references included, is evaluated on the same window** "
        f"({study.evaluation_start} to {study.evaluation_end}) -- the span for which "
        "out-of-sample forecasts exist. Scoring the references over the full sample while "
        "the models can only trade inside their test folds would compare windows, not "
        "strategies.",
        "",
    ]
    if not has_scenarios:
        lines += ["_No results._", ""]

    for scenario in ("frictionless", "realistic", "pessimistic") if has_scenarios else ():
        subset = study.economic_table[study.economic_table["scenario"] == scenario]
        if subset.empty:
            continue
        lines += [
            f"### {scenario.capitalize()}",
            "",
            table(
                subset[
                    [
                        "strategy",
                        "kind",
                        "annualised_return",
                        "annualised_volatility",
                        "sharpe_ratio",
                        "max_drawdown",
                        "annualised_turnover",
                        "n_trades",
                    ]
                ]
            ),
            "",
        ]

    lines += [
        "## 4. Data-snooping correction",
        "",
        "```json",
        json.dumps(study.snooping, indent=2, default=str),
        "```",
        "",
        "## Diagnostics",
        "",
        f"- Prices for `{study.stationarity['symbol']}`: **{study.stationarity['prices']}**",
        f"- Returns for `{study.stationarity['symbol']}`: **{study.stationarity['returns']}**",
        "",
        "A unit root in prices and stationarity in returns is the expected result and is a",
        "sanity check on the data, not a finding.",
        "",
        "## How to read this critically",
        "",
        "- A positive out-of-sample R-squared on daily returns above roughly 0.01 should be",
        "  treated as a bug until proven otherwise; look for look-ahead bias first.",
        "- The Sharpe ratios above are point estimates from one sample. Read the",
        "  bootstrap interval in section 4 to see their sampling uncertainty.",
        "- Every strategy here was evaluated on the same data. That is the multiple-testing",
        "  problem the deflated Sharpe ratio corrects for, and it is why the deflated number",
        "  matters more than the raw one.",
        "- Limitations that cannot be fixed with the available data are listed in",
        "  `docs/methodology.md`.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["COST_SCENARIOS", "StudyResult", "render_report", "run_research_study"]

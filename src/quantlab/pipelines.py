"""Reusable pipeline steps.

Everything an orchestrator needs to call lives here, as plain functions that
take arguments and return a small summary dict.  Airflow DAGs, the CLI and the
tests all call the same functions.

The rule this enforces: **no business logic in DAG files.**  Logic that lives in
a DAG can only be executed by Airflow, which means it cannot be unit tested,
cannot be run locally to debug, and cannot be reused by the API or a notebook.
A DAG here is a schedule and a dependency graph, nothing more.

Each function manages its own session and is idempotent, so a retried Airflow
task converges to the same database state as a successful first attempt.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from quantlab.config import get_settings
from quantlab.db import models as m
from quantlab.db import repository as repo
from quantlab.db.enums import AssetClass, CheckStatus
from quantlab.db.session import session_scope
from quantlab.exceptions import DataQualityError, QuantlabError
from quantlab.features.library import default_feature_set
from quantlab.features.pipeline import (
    build_feature_frame,
    build_training_frame,
    persist_feature_set,
    persist_feature_values,
    summarise_coverage,
)
from quantlab.ingestion.pipeline import ingest_economic, ingest_prices, materialise_returns
from quantlab.ingestion.registry import get_economic_provider, get_price_provider
from quantlab.logging import get_logger
from quantlab.timeutils import UTC, to_utc

log = get_logger(__name__)

#: The default demo universe: broad US equity, a bond proxy, gold and an
#: international index.  Liquid, long-history ETFs available from Stooq without a
#: key, and diverse enough that a portfolio built on them is not a single bet.
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "spy.us",
    "qqq.us",
    "iwm.us",
    "efa.us",
    "tlt.us",
    "gld.us",
    "xle.us",
    "xlf.us",
)

DEFAULT_MACRO_CODES: tuple[str, ...] = ("DGS10", "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2", "UNRATE")

DEFAULT_BENCHMARK = "SPY.US"


def _normalise(symbols: Sequence[str]) -> list[str]:
    return [s.strip() for s in symbols if s.strip()]


def resolve_universe(
    session: Session,
    symbols: Sequence[str] | None,
    *,
    universe: str | None = None,
    as_of: dt.datetime | None = None,
    context: str = "pipeline",
) -> list[str]:
    """Decide which symbols a step operates on, and be honest about how.

    Three paths, in order of preference:

    1.  An explicit symbol list.  The caller has decided; nothing to warn about.
    2.  A ``universe`` code.  Membership is resolved **as of** ``as_of`` through
        ``universe_members``, which is the survivorship-free answer.
    3.  Neither.  Falls back to every asset currently in the database, which is
        *today's* membership applied to the whole of history.  That is
        survivorship bias, and it is logged at WARNING every time rather than
        left implicit.

    An earlier version had only path 3.  The point-in-time membership tables and
    ``universe_symbols_as_of`` existed, were tested, and were called by nothing,
    so the documentation claimed a protection the pipelines did not use.
    """
    if symbols:
        return _normalise(symbols)
    if universe:
        cutoff = as_of or dt.datetime.now(tz=UTC)
        members = repo.universe_symbols_as_of(session, universe, cutoff)
        if not members:
            raise QuantlabError(
                "universe resolved to no members", universe=universe, as_of=str(cutoff)
            )
        log.info(
            "universe.resolved_point_in_time",
            universe=universe,
            as_of=str(cutoff),
            n=len(members),
        )
        return members

    resolved = [a.symbol for a in repo.list_assets(session)]
    log.warning(
        "universe.survivorship_fallback",
        context=context,
        n=len(resolved),
        detail=(
            "using every asset currently stored, which applies today's membership to "
            "all of history. Pass universe= for point-in-time membership. See "
            "docs/methodology.md#survivorship-bias."
        ),
    )
    return resolved


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def run_price_ingestion(
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    *,
    provider: str = "stooq",
    start: dt.date | None = None,
    end: dt.date | None = None,
    asset_class: AssetClass = AssetClass.ETF,
    fail_on_any_error: bool = False,
    **provider_kwargs: Any,
) -> dict[str, Any]:
    """Fetch, archive, validate and load daily bars."""
    symbols = _normalise(symbols)
    price_provider = get_price_provider(provider, **provider_kwargs)
    with session_scope() as session:
        summary = ingest_prices(
            session, price_provider, symbols, start=start, end=end, asset_class=asset_class
        )
        summary.raise_if_all_failed()
        if fail_on_any_error and summary.failed:
            raise QuantlabError("some symbols failed", failed=[r.symbol for r in summary.failed])
        payload = {
            "provider": provider,
            "requested": len(symbols),
            "succeeded": [r.symbol for r in summary.succeeded],
            "failed": {r.symbol: r.error for r in summary.failed},
            "rows_written": summary.rows_written,
        }
    log.info("pipeline.prices_ingested", **{k: v for k, v in payload.items() if k != "failed"})
    return payload


def run_macro_ingestion(
    codes: Sequence[str] = DEFAULT_MACRO_CODES,
    *,
    provider: str = "fred",
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> dict[str, Any]:
    codes = _normalise(codes)
    economic_provider = get_economic_provider(provider)
    with session_scope() as session:
        summary = ingest_economic(session, economic_provider, codes, start=start, end=end)
        summary.raise_if_all_failed()
        payload = {
            "provider": provider,
            "succeeded": [r.symbol for r in summary.succeeded],
            "failed": {r.symbol: r.error for r in summary.failed},
            "rows_written": summary.rows_written,
        }
    log.info("pipeline.macro_ingested", rows=payload["rows_written"])
    return payload


def run_return_materialisation(
    symbols: Sequence[str] | None = None,
    *,
    horizons: Sequence[int] = (1, 5, 21),
    universe: str | None = None,
) -> dict[str, Any]:
    with session_scope() as session:
        resolved = resolve_universe(
            session, symbols, universe=universe, context="return_materialisation"
        )
        written = materialise_returns(session, resolved, horizons=tuple(horizons))
    return {"symbols": len(resolved), "rows_written": written}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def run_data_validation(
    symbols: Sequence[str] | None = None,
    *,
    lookback_days: int = 30,
    max_stale_days: int = 7,
    fail_on_error: bool = True,
    universe: str | None = None,
) -> dict[str, Any]:
    """Post-load checks on what is actually in the database.

    Distinct from the checks run during ingestion: those validate a provider
    payload before it is written, these validate the warehouse afterwards.  The
    two catch different failures -- a provider can return perfect data that
    nonetheless never reaches the database because a load silently no-ops, and
    only a post-load check sees that.
    """
    now = dt.datetime.now(tz=UTC)
    issues: list[dict[str, Any]] = []
    checked: list[str] = []

    with session_scope() as session:
        resolved = resolve_universe(session, symbols, universe=universe, context="validation")
        rows: list[dict[str, Any]] = []
        for symbol in resolved:
            first, last, count = repo.price_coverage(session, symbol)
            checked.append(symbol)
            status = CheckStatus.PASSED
            message = f"{count} bars from {first} to {last}"
            if count == 0:
                status = CheckStatus.FAILED
                message = "no price data"
            elif last is not None and (now - last).days > max_stale_days:
                status = CheckStatus.WARNED
                message = f"latest bar is {(now - last).days} days old"
            if status is not CheckStatus.PASSED:
                issues.append({"symbol": symbol, "status": str(status), "message": message})
            rows.append(
                {
                    "dataset": "prices",
                    "entity": symbol,
                    "check_name": "warehouse_coverage",
                    "status": status,
                    "observed": {
                        "rows": count,
                        "first_ts": str(first),
                        "last_ts": str(last),
                        "lookback_days": lookback_days,
                    },
                    "message": message,
                    "checked_at": now,
                    "ingestion_run_id": None,
                }
            )
        repo.bulk_insert(session, m.DataQualityCheck, rows)

    failures = [i for i in issues if i["status"] == str(CheckStatus.FAILED)]
    if failures and fail_on_error:
        raise DataQualityError("warehouse validation failed", failures=failures[:5])
    log.info("pipeline.validated", checked=len(checked), issues=len(issues))
    return {"checked": checked, "issues": issues}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def assert_upstream_fresh(
    *,
    max_age_days: int = 5,
    symbols: Sequence[str] | None = None,
    universe: str | None = None,
    now: dt.datetime | None = None,
    context: str = "pipeline",
) -> dict[str, Any]:
    """Fail loudly if the price table has not been refreshed recently.

    Why this exists: the downstream DAGs (``feature_generation``,
    ``model_training``) are scheduled by cron a couple of hours after
    ``market_data_ingestion``, not wired to it by a dependency.  Cron does not
    know whether the upstream run succeeded.  If ingestion fails on Monday,
    feature generation still fires on Tuesday, recomputes the same features from
    Friday's prices, and reports success -- so the pipeline goes stale silently
    and every downstream number quietly describes the wrong week.

    A cross-DAG sensor would be the textbook answer, but it couples two
    schedules and deadlocks the downstream DAG whenever the upstream one is
    skipped or manually cleared.  Checking the actual state of the data is the
    more direct question: not "did a task succeed?" but "is the data I am about
    to model current?".  ``max_age_days`` defaults to 5 so that a long weekend
    plus one public holiday does not trip it.

    Raises :class:`DataQualityError` rather than returning a flag: a stale run
    must stop, not annotate itself.
    """
    reference = now or dt.datetime.now(tz=UTC)
    with session_scope() as session:
        resolved = resolve_universe(session, symbols, universe=universe, context=context)
        stamps: dict[str, dt.datetime] = {}
        for symbol in resolved:
            asset = repo.get_asset_by_symbol(session, symbol)
            if asset is None:
                continue
            ts = repo.latest_price_ts(session, asset.id)
            if ts is not None:
                stamps[symbol] = to_utc(ts)

    if not stamps:
        raise DataQualityError(
            "no prices found for any requested symbol -- upstream ingestion has never run",
            symbols=resolved[:10],
            n_symbols=len(resolved),
        )

    newest = max(stamps.values())
    age_days = (reference - newest).total_seconds() / 86_400.0
    stale = sorted(
        (
            sym
            for sym, ts in stamps.items()
            if (reference - ts).total_seconds() / 86_400.0 > max_age_days
        ),
    )
    if age_days > max_age_days:
        raise DataQualityError(
            "price data is stale -- upstream ingestion has not landed",
            newest_ts=str(newest),
            age_days=round(age_days, 2),
            max_age_days=max_age_days,
            context=context,
        )
    if stale:
        # The panel as a whole is current but individual names are not.  That is
        # a warning, not a failure: a delisted or halted ticker legitimately
        # stops printing while the rest of the universe keeps trading.
        log.warning(
            "pipeline.partial_staleness",
            context=context,
            n_stale=len(stale),
            stale=stale[:10],
            max_age_days=max_age_days,
        )
    log.info(
        "pipeline.upstream_fresh",
        context=context,
        newest_ts=str(newest),
        age_days=round(age_days, 2),
        n_symbols=len(stamps),
    )
    return {
        "fresh": True,
        "newest_ts": str(newest),
        "age_days": round(age_days, 2),
        "max_age_days": max_age_days,
        "n_symbols": len(stamps),
        "stale_symbols": stale,
    }


def run_feature_generation(
    symbols: Sequence[str] | None = None,
    *,
    include_macro: bool = True,
    macro_codes: Sequence[str] = DEFAULT_MACRO_CODES,
    persist: bool = True,
    universe: str | None = None,
) -> dict[str, Any]:
    feature_set = default_feature_set(
        include_macro=include_macro, macro_codes=list(macro_codes) if include_macro else None
    )
    with session_scope() as session:
        resolved = resolve_universe(session, symbols, universe=universe, context="features")
        prices = repo.load_prices(session, resolved)
        if prices.empty:
            raise QuantlabError("no prices available for feature generation", symbols=resolved)
        macro = repo.load_economic(session, list(macro_codes)) if include_macro else None

        frame = build_feature_frame(prices, feature_set, macro=macro, macro_codes=list(macro_codes))
        coverage = summarise_coverage(frame, feature_set.names)

        written = 0
        feature_set_id: int | None = None
        if persist:
            row = persist_feature_set(session, feature_set)
            feature_set_id = row.id
            written = persist_feature_values(session, row, frame, feature_names=feature_set.names)

    payload = {
        "feature_set": feature_set.name,
        "feature_set_id": feature_set_id,
        "spec_hash": feature_set.spec_hash(),
        "n_features": len(feature_set.names),
        "rows": int(len(frame)),
        "values_written": written,
        "worst_coverage": coverage.head(3).to_dict("records") if not coverage.empty else [],
    }
    log.info("pipeline.features_built", **{k: payload[k] for k in ("rows", "n_features")})
    return payload


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def run_model_training(
    symbols: Sequence[str] | None = None,
    *,
    model_specs: Sequence[tuple[str, dict[str, Any]]] | None = None,
    horizon: int = 1,
    include_macro: bool = True,
    macro_codes: Sequence[str] = DEFAULT_MACRO_CODES,
    n_splits: int = 5,
    test_size: int = 126,
    min_train_size: int = 756,
    embargo_days: int = 5,
    persist: bool = True,
    run_prefix: str = "scheduled",
    universe: str | None = None,
) -> dict[str, Any]:
    """Walk-forward training and evaluation of the model ladder."""
    from quantlab.models.registry import DEFAULT_REGRESSION_LADDER, get_model
    from quantlab.models.splitters import ExpandingWindowSplitter, SplitterConfig
    from quantlab.models.tracking import ExperimentTracker, frame_fingerprint
    from quantlab.models.walkforward import WalkForwardRunner

    settings = get_settings()
    specs = list(model_specs or DEFAULT_REGRESSION_LADDER)
    feature_set = default_feature_set(
        include_macro=include_macro, macro_codes=list(macro_codes) if include_macro else None
    )

    with session_scope() as session:
        resolved = resolve_universe(session, symbols, universe=universe, context="training")
        prices = repo.load_prices(session, resolved)
        if prices.empty:
            raise QuantlabError("no prices available for training", symbols=resolved)
        macro = repo.load_economic(session, list(macro_codes)) if include_macro else None
        X, y, meta = build_training_frame(prices, feature_set, macro=macro, horizon=horizon)

        splitter = ExpandingWindowSplitter(
            SplitterConfig(
                n_splits=n_splits,
                test_size=test_size,
                min_train_size=min_train_size,
                embargo=pd.Timedelta(days=embargo_days),
            )
        )
        runner = WalkForwardRunner(splitter)
        tracker = ExperimentTracker()
        fingerprint = frame_fingerprint(X)
        feature_set_row = persist_feature_set(session, feature_set) if persist else None

        results: list[dict[str, Any]] = []
        for name, params in specs:
            started = dt.datetime.now(tz=UTC)
            model = get_model(name, **params)
            result = runner.run(model, X, y, meta, seed=settings.random_seed)
            manifest = tracker.build_manifest(
                result,
                run_name=f"{run_prefix}_{name}",
                data_fingerprint=fingerprint,
                feature_set_hash=feature_set.spec_hash(),
                started_at=started,
            )
            tracker.write_manifest(manifest)
            model_run_id = None
            if persist:
                run_row = tracker.persist_run(
                    session,
                    result,
                    manifest,
                    feature_set_id=feature_set_row.id if feature_set_row else None,
                )
                model_run_id = run_row.id
            results.append(
                {
                    "model": name,
                    "model_run_id": model_run_id,
                    "config_hash": manifest.config_hash,
                    "metrics": result.overall.to_dict() if result.overall else None,
                }
            )

    log.info("pipeline.models_trained", models=len(results), rows=int(len(X)))
    return {
        "n_models": len(results),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "data_fingerprint": fingerprint,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
def run_backtests(
    symbols: Sequence[str] | None = None,
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    initial_cash: float = 1_000_000.0,
    commission_per_share: float = 0.005,
    slippage_bps: float = 5.0,
    rebalance: str = "weekly",
    persist: bool = True,
    universe: str | None = None,
) -> dict[str, Any]:
    """Run the classical strategy set and persist the results."""
    from quantlab.backtesting.costs import CostModel, FixedBpsSlippage, PerShareCommission
    from quantlab.backtesting.engine import BacktestEngine, EngineConfig
    from quantlab.backtesting.execution import ExecutionModel
    from quantlab.backtesting.market import MarketData
    from quantlab.backtesting.results import build_result, persist_result
    from quantlab.backtesting.sizing import TargetWeightSizer
    from quantlab.backtesting.strategy import BuyAndHold, MovingAverageCrossover, TimeSeriesMomentum

    with session_scope() as session:
        resolved = resolve_universe(session, symbols, universe=universe, context="backtesting")
        prices = repo.load_prices(session, resolved)
        if prices.empty:
            raise QuantlabError("no prices available for backtesting", symbols=resolved)
        market = MarketData(prices)

        costs = CostModel(
            commission=PerShareCommission(rate_per_share=commission_per_share),
            slippage=FixedBpsSlippage(slippage_bps),
        )
        config = EngineConfig(
            initial_cash=initial_cash,
            execution=ExecutionModel(costs=costs),
            sizer=TargetWeightSizer(allow_fractional=False),
            rebalance=rebalance,
            rebalance_tolerance=0.02,
            benchmark_symbol=benchmark if benchmark in market.symbols else None,
        )
        engine = BacktestEngine(config)

        strategies = [
            BuyAndHold(resolved),
            MovingAverageCrossover(resolved, fast=20, slow=100),
            TimeSeriesMomentum(resolved, lookback=126, allow_short=False),
        ]

        summaries: list[dict[str, Any]] = []
        for strategy in strategies:
            output = engine.run(strategy, market)
            result = build_result(output)
            backtest_id = None
            if persist:
                row = persist_result(
                    session,
                    result,
                    strategy_name=strategy.name,
                    strategy_kind=type(strategy).__name__,
                    strategy_params=strategy.describe(),
                    benchmark_symbol=config.benchmark_symbol,
                )
                backtest_id = row.id
            summaries.append(
                {
                    "strategy": strategy.name,
                    "backtest_id": backtest_id,
                    "metrics": result.metrics.to_dict(),
                }
            )

    log.info("pipeline.backtests_complete", n=len(summaries))
    return {"n_backtests": len(summaries), "results": summaries}


__all__ = [
    "DEFAULT_BENCHMARK",
    "DEFAULT_MACRO_CODES",
    "DEFAULT_SYMBOLS",
    "resolve_universe",
    "run_backtests",
    "run_data_validation",
    "run_feature_generation",
    "run_macro_ingestion",
    "run_model_training",
    "run_price_ingestion",
    "run_return_materialisation",
]

"""Load a deterministic synthetic dataset.

Used by ``docker compose --profile demo up`` and ``make seed`` so that a new
developer gets a working system with charts on it in one command, without a
network call or an API key.

Everything loaded here is synthetic and labelled as such: symbols are prefixed
``SYN_``, ``prices.source`` is ``"synthetic"``, and the generator parameters are
written into the raw manifest.  No synthetic number is ever presented as a real
market observation.

Three of the four series are unpredictable random walks.  One, ``SYN_TREND``,
has a small injected autocorrelation, so the demo shows the research pipeline
finding signal where signal exists and finding none where it does not -- which
is a more useful demonstration than four series that all look the same.
"""

from __future__ import annotations

import datetime as dt
import sys
from typing import Any

import pandas as pd

from quantlab.backtesting.costs import CostModel, FixedBpsSlippage, PerShareCommission
from quantlab.backtesting.engine import BacktestEngine, EngineConfig
from quantlab.backtesting.execution import ExecutionModel
from quantlab.backtesting.market import MarketData
from quantlab.backtesting.results import build_result, persist_result
from quantlab.backtesting.sizing import TargetWeightSizer
from quantlab.backtesting.strategy import BuyAndHold, MovingAverageCrossover, TimeSeriesMomentum
from quantlab.config import get_settings
from quantlab.db import repository as repo
from quantlab.db.enums import AssetClass
from quantlab.db.session import create_all, session_scope
from quantlab.features.library import default_feature_set
from quantlab.features.pipeline import (
    build_feature_frame,
    persist_feature_set,
    persist_feature_values,
)
from quantlab.ingestion.pipeline import ingest_prices, materialise_returns
from quantlab.ingestion.providers.synthetic import SyntheticProvider, SyntheticSpec
from quantlab.logging import configure_logging, get_logger

log = get_logger(__name__)

START = dt.date(2012, 1, 2)
END = dt.date(2023, 12, 29)

#: (symbol, annual drift, annual volatility, injected AR(1) coefficient)
DEMO_ASSETS: tuple[tuple[str, float, float, float], ...] = (
    ("SYN_BROAD", 0.07, 0.16, 0.00),
    ("SYN_GROWTH", 0.09, 0.24, 0.00),
    ("SYN_DEFENSIVE", 0.04, 0.09, 0.00),
    ("SYN_TREND", 0.06, 0.18, 0.20),
)

BENCHMARK = "SYN_BROAD"


def seed(create_schema: bool = False) -> dict[str, Any]:
    settings = get_settings()
    settings.ensure_directories()
    if create_schema:
        create_all()

    summary: dict[str, Any] = {"symbols": [], "backtests": []}

    with session_scope() as session:
        for symbol, drift, volatility, autocorrelation in DEMO_ASSETS:
            spec = SyntheticSpec(
                start=START,
                end=END,
                annual_drift=drift,
                annual_volatility=volatility,
                autocorrelation=autocorrelation,
                regime_switching=True,
            )
            provider = SyntheticProvider(spec, seed=settings.random_seed)
            result = ingest_prices(
                session, provider, [symbol], asset_class=AssetClass.ETF, start=START, end=END
            )
            result.raise_if_all_failed()
            summary["symbols"].append(symbol)
        session.flush()

        symbols = [a[0] for a in DEMO_ASSETS]
        summary["return_rows"] = materialise_returns(session, symbols, horizons=(1, 5, 21))

        prices = repo.load_prices(session, symbols)
        feature_set = default_feature_set(include_macro=False)
        frame = build_feature_frame(prices, feature_set)
        row = persist_feature_set(session, feature_set)
        summary["feature_values"] = persist_feature_values(
            session, row, frame, feature_names=feature_set.names
        )
        summary["feature_set"] = {"name": feature_set.name, "hash": feature_set.spec_hash()}

        market = MarketData(prices)
        config = EngineConfig(
            initial_cash=1_000_000.0,
            execution=ExecutionModel(costs=CostModel(PerShareCommission(), FixedBpsSlippage(5.0))),
            sizer=TargetWeightSizer(allow_fractional=False),
            rebalance="weekly",
            rebalance_tolerance=0.02,
            benchmark_symbol=BENCHMARK,
        )
        engine = BacktestEngine(config)
        for strategy in (
            BuyAndHold(symbols),
            MovingAverageCrossover(symbols, fast=20, slow=100),
            TimeSeriesMomentum(symbols, lookback=126, allow_short=False),
        ):
            output = engine.run(strategy, market)
            backtest_result = build_result(output)
            stored = persist_result(
                session,
                backtest_result,
                strategy_name=strategy.name,
                strategy_kind=type(strategy).__name__,
                strategy_params=strategy.describe(),
                benchmark_symbol=BENCHMARK,
            )
            summary["backtests"].append(
                {
                    "id": stored.id,
                    "strategy": strategy.name,
                    "sharpe": backtest_result.metrics.sharpe_ratio,
                    "total_return": backtest_result.metrics.total_return,
                }
            )

    log.info("demo.seeded", **{k: v for k, v in summary.items() if k != "backtests"})
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Load the synthetic demo dataset")
    parser.add_argument(
        "--create-schema",
        action="store_true",
        help="create tables directly instead of assuming migrations have run",
    )
    args = parser.parse_args(argv)

    configure_logging()
    summary = seed(create_schema=args.create_schema)
    print(pd.DataFrame(summary["backtests"]).to_string(index=False))
    print(
        f"\nSeeded {len(summary['symbols'])} synthetic assets, "
        f"{summary['feature_values']:,} feature values, "
        f"{len(summary['backtests'])} backtests."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

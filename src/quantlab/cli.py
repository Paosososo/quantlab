"""Command line interface.

Thin argparse wrapper over :mod:`quantlab.pipelines`.  Every command maps to
one pipeline function, so anything the CLI can do, Airflow can do, and both run
the same code.

argparse rather than Typer or Click: the CLI is a dozen subcommands over
functions that already exist, and adding a dependency for that is not a trade
worth making in a project whose install story matters.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from typing import Any

from quantlab import __version__
from quantlab.config import get_settings
from quantlab.logging import configure_logging, get_logger

log = get_logger(__name__)


def _date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


# Every handler takes the parsed namespace so the dispatcher can call them
# uniformly, even the few that need nothing from it.
# ruff: noqa: ARG001


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_init_db(args: argparse.Namespace) -> int:
    """Create the schema without Alembic.  Convenience for a throwaway database."""
    from quantlab.db.session import create_all, get_engine

    create_all()
    _emit({"created": True, "url": get_engine().url.render_as_string(hide_password=True)})
    return 0


def cmd_ingest_prices(args: argparse.Namespace) -> int:
    from quantlab.pipelines import DEFAULT_SYMBOLS, run_price_ingestion

    symbols = args.symbols or list(DEFAULT_SYMBOLS)
    kwargs: dict[str, Any] = {}
    if args.provider == "synthetic":
        from quantlab.ingestion.providers.synthetic import SyntheticSpec

        kwargs["spec"] = SyntheticSpec(
            start=args.start or dt.date(2010, 1, 4),
            end=args.end or dt.date.today(),
        )
        kwargs["seed"] = get_settings().random_seed
    _emit(
        run_price_ingestion(
            symbols, provider=args.provider, start=args.start, end=args.end, **kwargs
        )
    )
    return 0


def cmd_ingest_macro(args: argparse.Namespace) -> int:
    from quantlab.pipelines import DEFAULT_MACRO_CODES, run_macro_ingestion

    _emit(
        run_macro_ingestion(args.codes or list(DEFAULT_MACRO_CODES), start=args.start, end=args.end)
    )
    return 0


def cmd_returns(args: argparse.Namespace) -> int:
    from quantlab.pipelines import run_return_materialisation

    _emit(run_return_materialisation(args.symbols, horizons=args.horizons))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from quantlab.pipelines import run_data_validation

    payload = run_data_validation(args.symbols, fail_on_error=not args.warn_only)
    _emit(payload)
    return 1 if payload["issues"] and not args.warn_only else 0


def cmd_features(args: argparse.Namespace) -> int:
    from quantlab.pipelines import run_feature_generation

    _emit(run_feature_generation(args.symbols, include_macro=not args.no_macro))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from quantlab.pipelines import run_model_training

    _emit(
        run_model_training(
            args.symbols,
            horizon=args.horizon,
            include_macro=not args.no_macro,
            n_splits=args.n_splits,
            test_size=args.test_size,
            min_train_size=args.min_train_size,
            run_prefix=args.run_prefix,
        )
    )
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from quantlab.pipelines import run_backtests

    _emit(
        run_backtests(
            args.symbols,
            benchmark=args.benchmark,
            initial_cash=args.initial_cash,
            slippage_bps=args.slippage_bps,
            rebalance=args.rebalance,
        )
    )
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    """Run the full research question end to end and write the report."""
    from quantlab.research_run import run_research_study

    _emit(
        run_research_study(
            symbols=args.symbols, horizon=args.horizon, use_database=not args.offline
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from sqlalchemy import func, select

    from quantlab.db import models as m
    from quantlab.db.session import ping, session_scope

    if not ping():
        _emit({"database": "unreachable"})
        return 1
    with session_scope() as session:
        counts = {
            table.__tablename__: int(
                session.execute(select(func.count()).select_from(table.__table__)).scalar_one()
            )
            for table in (
                m.Asset,
                m.Price,
                m.Return,
                m.EconomicObservation,
                m.FeatureValue,
                m.ModelRun,
                m.Prediction,
                m.Backtest,
                m.Trade,
            )
        }
    _emit({"database": "ok", "version": __version__, "row_counts": counts})
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    _emit(get_settings().redacted())
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantlab",
        description="Financial data research and quantitative backtesting platform",
    )
    parser.add_argument("--version", action="version", version=f"quantlab {__version__}")
    parser.add_argument("--log-level", default=None, help="override QUANTLAB_LOG_LEVEL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.set_defaults(handler=handler)
        return sub

    add("init-db", cmd_init_db, "create all tables (development shortcut for Alembic)")
    add("status", cmd_status, "database connectivity and row counts")
    add("config", cmd_config, "print the resolved settings with secrets redacted")

    prices = add("ingest-prices", cmd_ingest_prices, "fetch and load daily bars")
    prices.add_argument("symbols", nargs="*", help="tickers; defaults to the demo universe")
    prices.add_argument("--provider", default="stooq", choices=["stooq", "synthetic"])
    prices.add_argument("--start", type=_date)
    prices.add_argument("--end", type=_date)

    macro = add("ingest-macro", cmd_ingest_macro, "fetch and load macroeconomic series")
    macro.add_argument("codes", nargs="*")
    macro.add_argument("--start", type=_date)
    macro.add_argument("--end", type=_date)

    returns = add("returns", cmd_returns, "materialise trailing and forward returns")
    returns.add_argument("symbols", nargs="*")
    returns.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 21])

    validate = add("validate", cmd_validate, "run warehouse data-quality checks")
    validate.add_argument("symbols", nargs="*")
    validate.add_argument("--warn-only", action="store_true", help="exit 0 even on failures")

    features = add("features", cmd_features, "build and store the feature set")
    features.add_argument("symbols", nargs="*")
    features.add_argument("--no-macro", action="store_true")

    train = add("train", cmd_train, "walk-forward training of the model ladder")
    train.add_argument("symbols", nargs="*")
    train.add_argument("--horizon", type=int, default=1)
    train.add_argument("--no-macro", action="store_true")
    train.add_argument("--n-splits", type=int, default=5)
    train.add_argument("--test-size", type=int, default=126)
    train.add_argument("--min-train-size", type=int, default=756)
    train.add_argument("--run-prefix", default="cli")

    backtest = add("backtest", cmd_backtest, "run the classical strategy set")
    backtest.add_argument("symbols", nargs="*")
    backtest.add_argument("--benchmark", default="SPY.US")
    backtest.add_argument("--initial-cash", type=float, default=1_000_000.0)
    backtest.add_argument("--slippage-bps", type=float, default=5.0)
    backtest.add_argument(
        "--rebalance", default="weekly", choices=["daily", "weekly", "monthly", "quarterly"]
    )

    research = add("research", cmd_research, "run the full research study and write the report")
    research.add_argument("symbols", nargs="*")
    research.add_argument("--horizon", type=int, default=1)
    research.add_argument(
        "--offline",
        action="store_true",
        help="use the synthetic generator instead of the database (no network, no PostgreSQL)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        log.warning("cli.interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

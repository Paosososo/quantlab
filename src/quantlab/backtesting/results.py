"""Backtest results: frames, metrics and persistence."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from quantlab.backtesting.engine import EngineOutput
from quantlab.backtesting.metrics import PerformanceMetrics, compute_metrics, drawdown_series
from quantlab.backtesting.types import Snapshot
from quantlab.db import models as m
from quantlab.db import repository as repo
from quantlab.db.enums import ExecutionTiming, RunStatus
from quantlab.logging import get_logger

log = get_logger(__name__)


def config_hash(payload: dict[str, Any]) -> str:
    """Stable content hash of a config dict.

    Sorted keys and a canonical separator so that the same configuration always
    hashes the same regardless of dict ordering.  This is what lets a run be
    matched to an earlier one without trusting a hand-written version label.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class BacktestResult:
    output: EngineOutput
    equity: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    metrics: PerformanceMetrics
    config: dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    @property
    def equity_curve(self) -> pd.Series:
        return self.equity["equity"]

    @property
    def benchmark_curve(self) -> pd.Series | None:
        if "benchmark_equity" not in self.equity.columns:
            return None
        series = self.equity["benchmark_equity"].dropna()
        return series if not series.empty else None

    @property
    def drawdown(self) -> pd.Series:
        return drawdown_series(self.equity_curve)

    def summary(self) -> dict[str, Any]:
        return self.metrics.to_dict()

    def to_json(self) -> str:
        return json.dumps({"config": self.config, "metrics": self.summary()}, indent=2, default=str)


def build_result(
    output: EngineOutput,
    *,
    periods_per_year: int | None = None,
    risk_free_annual: float = 0.0,
) -> BacktestResult:
    equity = _equity_frame(output)
    trades = _trades_frame(output)
    orders = _orders_frame(output)

    total_commission = float(trades["commission"].sum()) if not trades.empty else 0.0
    total_slippage = float(trades["slippage_cost"].sum()) if not trades.empty else 0.0

    metrics = compute_metrics(
        equity["equity"],
        turnover=equity.get("turnover"),
        gross_exposure=equity.get("gross_exposure"),
        benchmark_equity=equity.get("benchmark_equity"),
        trade_pnl=trades["realised_pnl"] if "realised_pnl" in trades.columns else None,
        periods_per_year=periods_per_year,
        risk_free_annual=risk_free_annual,
        total_commission=total_commission,
        total_slippage=total_slippage,
    )
    config = {
        "engine": output.config,
        "strategy": output.strategy,
        "symbols": output.symbols,
        "start_ts": output.start_ts.isoformat() if output.start_ts else None,
        "end_ts": output.end_ts.isoformat() if output.end_ts else None,
    }
    return BacktestResult(
        output=output, equity=equity, trades=trades, orders=orders, metrics=metrics, config=config
    )


def _equity_frame(output: EngineOutput) -> pd.DataFrame:
    if not output.snapshots:
        return pd.DataFrame(
            columns=[
                "cash",
                "positions_value",
                "equity",
                "gross_exposure",
                "net_exposure",
                "leverage",
                "n_positions",
                "period_return",
                "turnover",
                "benchmark_equity",
            ],
            index=pd.DatetimeIndex([], tz="UTC", name="ts"),
        )
    rows = [
        {
            "ts": s.ts,
            "cash": s.cash,
            "positions_value": s.positions_value,
            "equity": s.equity,
            "gross_exposure": s.gross_exposure,
            "net_exposure": s.net_exposure,
            "leverage": s.leverage,
            "n_positions": s.n_positions,
            "period_return": s.period_return,
            "turnover": s.turnover,
            "benchmark_equity": s.benchmark_equity,
        }
        for s in output.snapshots
    ]
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame = frame.set_index("ts").sort_index()
    frame["drawdown"] = drawdown_series(frame["equity"])
    return frame


def _trades_frame(output: EngineOutput) -> pd.DataFrame:
    if not output.fills:
        return pd.DataFrame(
            columns=[
                "symbol",
                "decision_ts",
                "execution_ts",
                "side",
                "quantity",
                "reference_price",
                "fill_price",
                "commission",
                "slippage_cost",
                "notional",
                "realised_pnl",
                "order_id",
            ]
        )
    realised = output.fill_realised_pnl or [float("nan")] * len(output.fills)
    rows = [
        {
            "order_id": f.order_id,
            "symbol": f.symbol,
            "decision_ts": f.decision_ts,
            "execution_ts": f.execution_ts,
            "side": str(f.side),
            "quantity": f.quantity,
            "reference_price": f.reference_price,
            "fill_price": f.fill_price,
            "commission": f.commission,
            "slippage_cost": f.slippage_cost,
            "notional": f.notional,
            "realised_pnl": realised[i] if i < len(realised) else float("nan"),
        }
        for i, f in enumerate(output.fills)
    ]
    frame = pd.DataFrame(rows)
    frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True)
    frame["execution_ts"] = pd.to_datetime(frame["execution_ts"], utc=True)
    return frame


def _orders_frame(output: EngineOutput) -> pd.DataFrame:
    if not output.orders:
        return pd.DataFrame(
            columns=[
                "order_id",
                "symbol",
                "side",
                "quantity",
                "decision_ts",
                "execution_ts",
                "status",
                "reject_reason",
                "tag",
            ]
        )
    rows = [
        {
            "order_id": r.order.order_id,
            "symbol": r.order.symbol,
            "side": str(r.order.side),
            "quantity": r.order.quantity,
            "decision_ts": r.order.decision_ts,
            "execution_ts": r.execution_ts,
            "status": str(r.status),
            "reject_reason": r.reject_reason,
            "tag": r.order.tag,
        }
        for r in output.orders
    ]
    frame = pd.DataFrame(rows)
    frame["decision_ts"] = pd.to_datetime(frame["decision_ts"], utc=True)
    frame["execution_ts"] = pd.to_datetime(frame["execution_ts"], utc=True)
    return frame


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def persist_result(
    session: Session,
    result: BacktestResult,
    *,
    strategy_name: str,
    strategy_kind: str,
    strategy_params: dict[str, Any],
    model_run_id: int | None = None,
    benchmark_symbol: str | None = None,
    code_version: str | None = None,
    store_positions: bool = False,
    replace_existing: bool = True,
) -> m.Backtest:
    """Write a completed backtest to the database.

    Orders are written first so that trades can reference them, which is what
    makes the ``execution_ts > decision_ts`` constraint enforceable at the
    database level rather than only in Python.

    ``replace_existing`` makes the write idempotent: a backtest is identified by
    its config hash, which covers the strategy parameters, the engine
    configuration, the symbol set and the date range.  Re-running the same
    configuration replaces the earlier result instead of appending a duplicate.
    That matters beyond tidiness -- ``/backtests/{id}/performance`` uses the
    number of stored backtests as the trial count for the deflated Sharpe ratio,
    so duplicate rows would inflate the correction and understate significance.
    """
    params_hash = config_hash(strategy_params)
    strategy = (
        session.query(m.Strategy)
        .filter(m.Strategy.name == strategy_name, m.Strategy.params_hash == params_hash)
        .one_or_none()
    )
    if strategy is None:
        strategy = m.Strategy(
            name=strategy_name, kind=strategy_kind, params=strategy_params, params_hash=params_hash
        )
        session.add(strategy)
        session.flush()

    run_hash = config_hash(result.config)
    if replace_existing:
        duplicates = session.query(m.Backtest).filter(m.Backtest.config_hash == run_hash).all()
        for row in duplicates:
            log.info("backtest.replacing_identical_run", backtest_id=row.id)
            session.delete(row)  # orders, trades and snapshots cascade
        if duplicates:
            session.flush()

    benchmark_asset = (
        repo.get_asset_by_symbol(session, benchmark_symbol) if benchmark_symbol else None
    )
    engine_cfg = result.config.get("engine", {})
    timing = engine_cfg.get("execution", {}).get("timing", str(ExecutionTiming.NEXT_OPEN))

    backtest = m.Backtest(
        strategy_id=strategy.id,
        model_run_id=model_run_id,
        benchmark_asset_id=benchmark_asset.id if benchmark_asset else None,
        start_ts=result.output.start_ts,
        end_ts=result.output.end_ts,
        initial_cash=repo.to_decimal(engine_cfg.get("initial_cash", 0.0)),
        execution_timing=ExecutionTiming(timing),
        cost_config=engine_cfg.get("execution", {}).get("costs", {}),
        engine_config=engine_cfg,
        config_hash=run_hash,
        code_version=code_version,
        status=RunStatus.SUCCEEDED,
        metrics=result.metrics.to_dict(),
    )
    session.add(backtest)
    session.flush()

    asset_ids = _asset_id_map(session, result)

    order_ids: dict[str, int] = {}
    for record in result.output.orders:
        asset_id = asset_ids.get(record.order.symbol)
        if asset_id is None:
            continue
        order_row = m.Order(
            backtest_id=backtest.id,
            asset_id=asset_id,
            client_order_id=record.order.order_id,
            decision_ts=record.order.decision_ts,
            execution_ts=record.execution_ts,
            side=record.order.side,
            order_type=record.order.order_type,
            quantity=repo.to_decimal(record.order.quantity, places=10),
            limit_price=repo.to_decimal(record.order.limit_price),
            status=record.status,
            reject_reason=record.reject_reason,
        )
        session.add(order_row)
        session.flush()
        order_ids[record.order.order_id] = order_row.id

    realised = result.output.fill_realised_pnl
    for i, fill in enumerate(result.output.fills):
        asset_id = asset_ids.get(fill.symbol)
        if asset_id is None:
            continue
        session.add(
            m.Trade(
                backtest_id=backtest.id,
                order_id=order_ids.get(fill.order_id),
                asset_id=asset_id,
                decision_ts=fill.decision_ts,
                execution_ts=fill.execution_ts,
                side=fill.side,
                quantity=repo.to_decimal(fill.quantity, places=10),
                reference_price=repo.to_decimal(fill.reference_price),
                fill_price=repo.to_decimal(fill.fill_price),
                commission=repo.to_decimal(fill.commission),
                slippage_cost=repo.to_decimal(fill.slippage_cost),
                notional=repo.to_decimal(fill.notional),
                realised_pnl=repo.to_decimal(realised[i]) if i < len(realised) else None,
            )
        )

    for snapshot in result.output.snapshots:
        session.add(
            m.PortfolioSnapshot(
                backtest_id=backtest.id,
                ts=snapshot.ts,
                cash=repo.to_decimal(snapshot.cash),
                positions_value=repo.to_decimal(snapshot.positions_value),
                equity=repo.to_decimal(snapshot.equity),
                gross_exposure=repo.to_decimal(snapshot.gross_exposure),
                net_exposure=repo.to_decimal(snapshot.net_exposure),
                leverage=float(snapshot.leverage),
                n_positions=int(snapshot.n_positions),
                period_return=float(snapshot.period_return),
                drawdown=None,
                benchmark_equity=repo.to_decimal(snapshot.benchmark_equity),
                turnover=float(snapshot.turnover),
            )
        )
        if store_positions and snapshot.holdings:
            # Marks for this bar, recovered from the engine's own snapshot rather
            # than invented.  An earlier version wrote price=0 and
            # market_value=0 for every holding, which is fabricated data in a
            # production table: it satisfies the schema, reads as a real number
            # downstream, and is wrong.  If the marks cannot be recovered the
            # rows are skipped and the omission is logged.
            marks = _marks_from_snapshot(snapshot)
            if marks is None:
                log.warning(
                    "backtest.position_snapshot_skipped",
                    ts=str(snapshot.ts),
                    reason="per-asset marks are not recoverable from the snapshot",
                )
            else:
                for symbol, quantity in snapshot.holdings.items():
                    asset_id = asset_ids.get(symbol)
                    price = marks.get(symbol)
                    if asset_id is None or price is None:
                        continue
                    market_value = quantity * price
                    session.add(
                        m.PositionSnapshot(
                            backtest_id=backtest.id,
                            asset_id=asset_id,
                            ts=snapshot.ts,
                            quantity=repo.to_decimal(quantity, places=10),
                            price=repo.to_decimal(price),
                            market_value=repo.to_decimal(market_value),
                            weight=float(market_value / snapshot.equity)
                            if snapshot.equity
                            else None,
                        )
                    )

    session.flush()
    log.info(
        "backtest.persisted",
        backtest_id=backtest.id,
        trades=len(result.output.fills),
        snapshots=len(result.output.snapshots),
    )
    return backtest


def _marks_from_snapshot(snapshot: Snapshot) -> dict[str, float] | None:
    """Per-asset marks for a snapshot, or None when they were not recorded.

    Returns None rather than a zero-filled dictionary: a caller that cannot get
    real marks must know that, not receive plausible-looking zeros.  Marks are
    only populated when the engine ran with ``record_holdings=True``.
    """
    if not snapshot.marks:
        return None
    return {str(k): float(v) for k, v in snapshot.marks.items()}


def _asset_id_map(session: Session, result: BacktestResult) -> dict[str, int]:
    symbols = set(result.output.symbols)
    symbols.update(result.trades["symbol"].unique() if not result.trades.empty else [])
    out: dict[str, int] = {}
    for symbol in symbols:
        asset = repo.get_asset_by_symbol(session, symbol)
        if asset is not None:
            out[symbol] = asset.id
        else:
            log.warning("backtest.persist.unknown_symbol", symbol=symbol)
    return out


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


__all__ = ["BacktestResult", "build_result", "config_hash", "persist_result"]

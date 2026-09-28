"""Backtests, equity curves, trades and performance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Path, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from quantlab.api.deps import Pagination, PaginationDep, SessionDep
from quantlab.api.errors import NotFoundError
from quantlab.api.schemas import (
    BacktestOut,
    EquityPoint,
    Page,
    PerformanceSummary,
    TradeOut,
)
from quantlab.db import models as m

router = APIRouter(prefix="/backtests", tags=["backtests"])


def _to_out(row: m.Backtest, session: Session) -> BacktestOut:
    strategy = session.get(m.Strategy, row.strategy_id)
    benchmark = session.get(m.Asset, row.benchmark_asset_id) if row.benchmark_asset_id else None
    return BacktestOut(
        id=row.id,
        strategy_name=strategy.name if strategy else "unknown",
        strategy_kind=strategy.kind if strategy else "unknown",
        start_ts=row.start_ts,
        end_ts=row.end_ts,
        initial_cash=float(row.initial_cash),
        execution_timing=str(row.execution_timing),
        config_hash=row.config_hash,
        status=str(row.status),
        benchmark_symbol=benchmark.symbol if benchmark else None,
        metrics=row.metrics,
    )


@router.get("", response_model=Page[BacktestOut])
def list_backtests(
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
    strategy: str | None = Query(default=None),
) -> Page[BacktestOut]:
    stmt = select(m.Backtest).order_by(m.Backtest.created_at.desc())
    if strategy:
        stmt = stmt.join(m.Strategy, m.Strategy.id == m.Backtest.strategy_id).where(
            m.Strategy.name == strategy
        )
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all()
    items = [_to_out(r, session) for r in rows]
    return Page[BacktestOut](
        items=items, limit=page.limit, offset=page.offset, returned=len(items), total=int(total)
    )


@router.get("/{backtest_id}", response_model=BacktestOut)
def get_backtest(backtest_id: int = Path(ge=1), session: Session = SessionDep) -> BacktestOut:
    row = session.get(m.Backtest, backtest_id)
    if row is None:
        raise NotFoundError("backtest not found", backtest_id=backtest_id)
    return _to_out(row, session)


@router.get("/{backtest_id}/equity", response_model=Page[EquityPoint])
def get_equity_curve(
    backtest_id: int = Path(ge=1),
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
) -> Page[EquityPoint]:
    if session.get(m.Backtest, backtest_id) is None:
        raise NotFoundError("backtest not found", backtest_id=backtest_id)

    stmt = (
        select(m.PortfolioSnapshot)
        .where(m.PortfolioSnapshot.backtest_id == backtest_id)
        .order_by(m.PortfolioSnapshot.ts)
    )
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all()

    equity = pd.Series(
        [float(r.equity) for r in rows], index=pd.DatetimeIndex([r.ts for r in rows])
    )
    drawdown = (equity / equity.cummax() - 1.0) if not equity.empty else equity

    items = [
        EquityPoint(
            ts=row.ts,
            equity=float(row.equity),
            cash=float(row.cash),
            positions_value=float(row.positions_value),
            period_return=row.period_return,
            # Computed on read rather than stored: the stored column would be
            # wrong for any page that does not start at the beginning of the run.
            drawdown=float(drawdown.iloc[i]) if not drawdown.empty else None,
            benchmark_equity=None if row.benchmark_equity is None else float(row.benchmark_equity),
            leverage=float(row.leverage),
            n_positions=int(row.n_positions),
            turnover=row.turnover,
        )
        for i, row in enumerate(rows)
    ]
    return Page[EquityPoint](
        items=items, limit=page.limit, offset=page.offset, returned=len(items), total=int(total)
    )


@router.get("/{backtest_id}/trades", response_model=Page[TradeOut])
def get_trades(
    backtest_id: int = Path(ge=1),
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
    symbol: str | None = Query(default=None),
) -> Page[TradeOut]:
    if session.get(m.Backtest, backtest_id) is None:
        raise NotFoundError("backtest not found", backtest_id=backtest_id)

    stmt = (
        select(m.Trade, m.Asset.symbol)
        .join(m.Asset, m.Asset.id == m.Trade.asset_id)
        .where(m.Trade.backtest_id == backtest_id)
        .order_by(m.Trade.execution_ts)
    )
    if symbol:
        stmt = stmt.where(m.Asset.symbol == symbol.upper())
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.limit(page.limit).offset(page.offset)).all()
    items = [
        TradeOut(
            symbol=sym,
            decision_ts=trade.decision_ts,
            execution_ts=trade.execution_ts,
            side=str(trade.side),
            quantity=float(trade.quantity),
            reference_price=float(trade.reference_price),
            fill_price=float(trade.fill_price),
            commission=float(trade.commission),
            slippage_cost=float(trade.slippage_cost),
            notional=float(trade.notional),
            realised_pnl=None if trade.realised_pnl is None else float(trade.realised_pnl),
        )
        for trade, sym in rows
    ]
    return Page[TradeOut](
        items=items, limit=page.limit, offset=page.offset, returned=len(items), total=int(total)
    )


@router.get("/{backtest_id}/performance", response_model=PerformanceSummary)
def get_performance(
    backtest_id: int = Path(ge=1),
    session: Session = SessionDep,
    n_trials: int | None = Query(
        default=None,
        ge=1,
        le=10_000,
        description=(
            "How many strategy configurations were evaluated before this one was "
            "selected.  Omit it and the server counts the backtests it has stored, "
            "which is a lower bound on the real number of trials."
        ),
    ),
) -> PerformanceSummary:
    """Performance, risk and Sharpe inference for one backtest.

    The Sharpe inference block is the part worth reading: it reports the
    probabilistic Sharpe ratio, the deflated Sharpe ratio, and how long a track
    record would have to be before the result was statistically distinguishable
    from zero.

    The deflation uses the distribution of Sharpe ratios across **every backtest
    in this database** as the trial distribution.  That is the honest thing to
    use: those are the configurations that were actually tried.  It is still a
    lower bound, because configurations abandoned before being persisted do not
    appear, and the docstring says so rather than pretending otherwise.
    """
    backtest = session.get(m.Backtest, backtest_id)
    if backtest is None:
        raise NotFoundError("backtest not found", backtest_id=backtest_id)

    rows = (
        session.execute(
            select(m.PortfolioSnapshot)
            .where(m.PortfolioSnapshot.backtest_id == backtest_id)
            .order_by(m.PortfolioSnapshot.ts)
        )
        .scalars()
        .all()
    )
    strategy = session.get(m.Strategy, backtest.strategy_id)

    equity = pd.Series(
        [float(r.equity) for r in rows], index=pd.DatetimeIndex([r.ts for r in rows])
    )
    # ``fill_method=None`` -- see quantlab.backtesting.metrics.to_returns.
    returns = equity.pct_change(fill_method=None).dropna()

    risk_payload: dict[str, object] = {}
    inference_payload: dict[str, object] | None = None
    if len(returns) >= 20:
        from quantlab.risk.metrics import summarise_risk

        risk_payload = summarise_risk(returns, equity=equity).to_dict()

    if len(returns) >= 30:
        from quantlab.research.multiple_testing import analyse_sharpe

        observed = _stored_sharpe_ratios(session)
        trials = max(n_trials or 0, len(observed), 1)
        trial_sharpes = observed if len(observed) > 1 else None
        inference_payload = analyse_sharpe(
            returns.to_numpy(),
            n_trials=trials if trials > 1 else None,
            trial_sharpes=trial_sharpes,
        ).to_dict()
        if inference_payload is not None:
            inference_payload["trial_source"] = (
                "stored backtests" if trial_sharpes is not None else "caller-declared"
            )

    return PerformanceSummary(
        backtest_id=backtest_id,
        strategy_name=strategy.name if strategy else "unknown",
        metrics=backtest.metrics or {},
        risk=risk_payload,
        sharpe_inference=inference_payload,
    )


def _stored_sharpe_ratios(session: Session) -> np.ndarray:
    """Per-period Sharpe ratios of every backtest held in the database.

    Converted from annualised to per-period, because that is the scale the
    deflated-Sharpe formulas work in.
    """
    rows: Sequence[Any] = (
        session.execute(select(m.Backtest.metrics).where(m.Backtest.metrics.is_not(None)))
        .scalars()
        .all()
    )
    values = [
        float(metrics["sharpe_ratio"]) / np.sqrt(252.0)
        for metrics in rows
        if isinstance(metrics, dict) and metrics.get("sharpe_ratio") is not None
    ]
    return np.asarray(values, dtype="float64")


__all__ = ["router"]

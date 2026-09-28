"""The event-driven backtest loop.

Bar ordering
------------
For each bar ``i`` on the timeline, in this exact order:

1.  **Execute** orders decided at bar ``i-1`` (or earlier) against bar ``i``.
    With ``NEXT_OPEN`` timing they fill at bar ``i``'s open.
2.  **Mark to market** at bar ``i``'s close and record a snapshot.
3.  **Decide**, using a point-in-time view built at bar ``i``'s availability
    instant.  New orders are queued for bar ``i+1``.

Steps 1 and 3 are separated by step 2 for a reason: it makes it structurally
impossible for a decision taken at bar ``i`` to be filled at bar ``i``'s price.
The queue only ever moves forward.

Why event-driven rather than vectorised
---------------------------------------
A vectorised backtest -- ``positions = signal.shift(1); pnl = positions *
returns`` -- is faster and shorter, and it is how most tutorials do it.  It is
also where look-ahead bugs hide, because the correctness of the whole thing
rests on a single ``shift`` whose sign nobody can verify by reading it.  The
loop below is slower, but every price used is fetched from a named bar at an
explicit timestamp, and a reader can follow one order from decision to fill.
For a project whose entire claim is temporal correctness, auditability beats
speed.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.backtesting.execution import ExecutionModel, Rejection
from quantlab.backtesting.market import MarketData
from quantlab.backtesting.portfolio import Portfolio
from quantlab.backtesting.sizing import PositionSizer, TargetWeightSizer
from quantlab.backtesting.strategy import (
    DecisionContext,
    PortfolioView,
    PredictionBook,
    Strategy,
)
from quantlab.backtesting.types import (
    Bar,
    Fill,
    Order,
    OrderRecord,
    Snapshot,
    TargetWeight,
    next_order_id,
)
from quantlab.db.enums import OrderSide, OrderStatus
from quantlab.exceptions import BacktestError, InvalidOrderError
from quantlab.logging import get_logger
from quantlab.pit import PointInTimeFrame
from quantlab.timeutils import to_utc

log = get_logger(__name__)

REBALANCE_FREQUENCIES = ("daily", "weekly", "monthly", "quarterly")


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Everything that changes engine behaviour, in one serialisable object."""

    initial_cash: float = 100_000.0
    execution: ExecutionModel = ExecutionModel()
    sizer: PositionSizer = TargetWeightSizer()
    #: How often the strategy is allowed to change its target weights.
    rebalance: str = "daily"
    #: Skip a rebalance whose weight change is smaller than this, to avoid
    #: paying costs for noise.  0.0 rebalances on any difference.
    rebalance_tolerance: float = 0.0
    #: Skip orders below this notional; brokers have minimums and tiny orders
    #: are dominated by fixed costs.
    min_trade_notional: float = 0.0
    #: Hard cap on gross exposure as a multiple of equity.
    max_gross_leverage: float = 1.0
    #: Cash may fall to ``-max_borrow_fraction * equity``.  0.0 forbids borrowing.
    max_borrow_fraction: float = 0.0
    #: Lookback used for the point-in-time volatility estimate fed to the sizer.
    volatility_lookback: int = 63
    #: Price field used for marking positions and computing returns.
    price_field: str = "adj_close"
    #: Store per-asset holdings in every snapshot (larger output, richer analysis).
    record_holdings: bool = True
    benchmark_symbol: str | None = None

    def __post_init__(self) -> None:
        if self.rebalance not in REBALANCE_FREQUENCIES:
            raise BacktestError(
                "unknown rebalance frequency",
                value=self.rebalance,
                allowed=list(REBALANCE_FREQUENCIES),
            )
        if self.initial_cash <= 0:
            raise BacktestError("initial cash must be positive", value=self.initial_cash)
        if self.max_gross_leverage <= 0:
            raise BacktestError("max gross leverage must be positive")

    def describe(self) -> dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "execution": self.execution.describe(),
            "sizer": self.sizer.describe(),
            "rebalance": self.rebalance,
            "rebalance_tolerance": self.rebalance_tolerance,
            "min_trade_notional": self.min_trade_notional,
            "max_gross_leverage": self.max_gross_leverage,
            "max_borrow_fraction": self.max_borrow_fraction,
            "volatility_lookback": self.volatility_lookback,
            "price_field": self.price_field,
            "benchmark_symbol": self.benchmark_symbol,
        }


@dataclass(slots=True)
class EngineOutput:
    """Raw output of one run.  Turned into metrics by :mod:`quantlab.backtesting.results`."""

    snapshots: list[Snapshot] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    #: Realised P&L from each fill, aligned index-for-index with ``fills``.
    fill_realised_pnl: list[float] = field(default_factory=list)
    orders: list[OrderRecord] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    start_ts: dt.datetime | None = None
    end_ts: dt.datetime | None = None
    symbols: list[str] = field(default_factory=list)


def _period_key(ts: pd.Timestamp, frequency: str) -> tuple[int, ...]:
    iso = ts.isocalendar()
    if frequency == "daily":
        return (ts.year, ts.month, ts.day)
    if frequency == "weekly":
        return (int(iso.year), int(iso.week))
    if frequency == "monthly":
        return (ts.year, ts.month)
    return (ts.year, (ts.month - 1) // 3)


class BacktestEngine:
    """Runs one strategy over one market data set."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    # -- public API --------------------------------------------------------
    def run(
        self,
        strategy: Strategy,
        market: MarketData,
        *,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        predictions: pd.DataFrame | None = None,
        features: Mapping[str, pd.DataFrame] | None = None,
    ) -> EngineOutput:
        cfg = self.config
        timeline = list(market.iter_timeline(start, end))
        if len(timeline) < 2:
            raise BacktestError("need at least two bars to run a backtest", bars=len(timeline))

        # Bars of history that exist *before* the run window.  A strategy with a
        # 100-bar warm-up run on a window preceded by ten years of data can
        # trade from bar one: the history it needs is loaded and visible in the
        # point-in-time view.  Counting warm-up from the window's own start
        # instead would make a strategy's behaviour depend on where the caller
        # happened to slice, which is how a walk-forward comparison ends up
        # measuring the slice rather than the strategy.
        history_offset = int(market.timeline.searchsorted(timeline[0], side="left"))

        portfolio = Portfolio(initial_cash=cfg.initial_cash)
        output = EngineOutput(
            config=cfg.describe(),
            strategy=strategy.describe(),
            start_ts=timeline[0].to_pydatetime(),
            end_ts=timeline[-1].to_pydatetime(),
            symbols=market.symbols,
        )

        prediction_book = (
            PredictionBook(predictions)
            if predictions is not None and not predictions.empty
            else None
        )
        if prediction_book is not None:
            log.info("backtest.predictions_loaded", n=len(prediction_book))

        pending: list[Order] = []
        records: dict[str, OrderRecord] = {}
        previous_equity = cfg.initial_cash
        last_rebalance_key: tuple[int, ...] | None = None
        benchmark = _BenchmarkTracker(cfg.benchmark_symbol, cfg.initial_cash, cfg.execution)

        for index, ts in enumerate(timeline):
            bars = market.bars_at(ts)

            # --- 1. execute orders decided strictly earlier -----------------
            traded_notional_before = portfolio.traded_notional
            still_pending: list[Order] = []
            for order in pending:
                bar = bars.get(order.symbol)
                if bar is None:
                    # The symbol did not trade today.  Keep the order alive for
                    # the next session rather than silently dropping the
                    # strategy's intent.
                    still_pending.append(order)
                    continue
                volatility = self._point_in_time_volatility(market, order.symbol, ts, cfg)
                outcome = cfg.execution.execute(order, bar, volatility=volatility)
                record = records[order.order_id]
                if isinstance(outcome, Rejection):
                    record.status = OrderStatus.REJECTED
                    record.reject_reason = outcome.reason
                    output.rejections.append(outcome)
                    log.debug("order.rejected", symbol=order.symbol, reason=outcome.reason)
                    continue
                realised = portfolio.apply_fill(outcome)
                output.fill_realised_pnl.append(realised)
                record.status = OrderStatus.FILLED
                record.execution_ts = outcome.execution_ts
                record.fill = outcome
                output.fills.append(outcome)
            pending = still_pending

            # --- 2. mark to market and snapshot ----------------------------
            marks = market.last_known_closes(ts, cfg.price_field)
            equity = portfolio.assert_accounting_identity(marks, label=str(ts))
            period_return = (equity / previous_equity - 1.0) if previous_equity else 0.0
            turnover = (
                (portfolio.traded_notional - traded_notional_before) / previous_equity
                if previous_equity
                else 0.0
            )
            benchmark.update(bars, marks)
            gross = portfolio.gross_exposure(marks)
            holdings = portfolio.holdings() if cfg.record_holdings else {}
            output.snapshots.append(
                Snapshot(
                    ts=ts.to_pydatetime(),
                    cash=portfolio.cash,
                    positions_value=portfolio.positions_value(marks),
                    equity=equity,
                    gross_exposure=gross,
                    net_exposure=portfolio.net_exposure(marks),
                    leverage=gross / equity if equity else 0.0,
                    n_positions=portfolio.n_positions,
                    period_return=period_return,
                    turnover=turnover,
                    benchmark_equity=benchmark.equity,
                    holdings=holdings,
                    marks={s: marks[s] for s in holdings if s in marks},
                )
            )
            previous_equity = equity

            # --- 3. decide, for execution at the NEXT bar -------------------
            if index == len(timeline) - 1:
                break  # nothing left to execute against
            if index + 1 + history_offset < strategy.warmup_bars():
                continue
            period_key = _period_key(ts, cfg.rebalance)
            if last_rebalance_key is not None and period_key == last_rebalance_key:
                continue

            as_of = self._decision_instant(bars, ts)
            context = DecisionContext(
                ts=ts.to_pydatetime(),
                as_of=as_of,
                prices=market.point_in_time(as_of),
                portfolio=PortfolioView(
                    equity=equity,
                    cash=portfolio.cash,
                    weights=portfolio.weights(marks),
                    quantities=portfolio.holdings(),
                ),
                features=self._point_in_time_features(features, as_of),
                predictions=prediction_book,
            )
            targets = strategy.on_bar(context)
            new_orders = self._targets_to_orders(
                targets, portfolio, marks, bars, equity, ts.to_pydatetime(), market, cfg
            )
            # Advance the schedule because a rebalance was *considered*, not
            # because one produced trades.  An earlier version only advanced it
            # on a fill, which meant a strategy whose targets already matched its
            # holdings was re-consulted every single bar -- "monthly"
            # rebalancing consulted the strategy 177 times in 200 bars.  The
            # trades then happened on whichever day the drift first became large
            # enough, so the trading calendar was data-dependent in a way the
            # configuration did not express.
            last_rebalance_key = period_key
            for order in new_orders:
                records[order.order_id] = OrderRecord(order=order)
                output.orders.append(records[order.order_id])
            pending.extend(new_orders)

        log.info(
            "backtest.complete",
            bars=len(output.snapshots),
            fills=len(output.fills),
            rejections=len(output.rejections),
            final_equity=round(previous_equity, 2),
        )
        return output

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _decision_instant(bars: Mapping[str, Bar], ts: pd.Timestamp) -> dt.datetime:
        """The instant the decision is taken: the latest close among today's bars.

        Using the *latest* availability across symbols is the conservative
        choice for a multi-market portfolio: it means a decision is only taken
        once every market involved has closed, rather than assuming a US
        strategy can react to a Tokyo close it has not yet seen.
        """
        if bars:
            return max(bar.available_at for bar in bars.values())
        return to_utc(ts.to_pydatetime())

    @staticmethod
    def _point_in_time_features(
        features: Mapping[str, pd.DataFrame] | None, as_of: dt.datetime
    ) -> dict[str, PointInTimeFrame]:
        if not features:
            return {}
        return {symbol: PointInTimeFrame(frame, as_of) for symbol, frame in features.items()}

    @staticmethod
    def _point_in_time_volatility(
        market: MarketData, symbol: str, ts: pd.Timestamp, cfg: EngineConfig
    ) -> float | None:
        """Trailing daily return volatility using bars strictly before ``ts``.

        Strictly before, because this estimate feeds the slippage model for an
        order being filled *at* ``ts``; including that bar's own move would be a
        one-day look-ahead inside the cost model.
        """
        return market.trailing_volatility(
            symbol,
            ts.to_pydatetime(),
            lookback=cfg.volatility_lookback,
            field=cfg.price_field,
            inclusive=False,
            minimum=5,
        )

    def _targets_to_orders(
        self,
        targets: Sequence[TargetWeight],
        portfolio: Portfolio,
        marks: Mapping[str, float],
        bars: Mapping[str, Bar],
        equity: float,
        decision_ts: dt.datetime,
        market: MarketData,
        cfg: EngineConfig,
    ) -> list[Order]:
        if not targets or equity <= 0:
            return []

        current_weights = portfolio.weights(marks)
        gross_target = sum(abs(t.weight) for t in targets)
        scale = 1.0
        if gross_target > cfg.max_gross_leverage > 0:
            scale = cfg.max_gross_leverage / gross_target
            log.debug("targets.scaled_for_leverage", gross=gross_target, scale=round(scale, 4))

        orders: list[Order] = []
        for target in targets:
            symbol = target.symbol
            if not math.isfinite(target.weight):
                # A NaN weight is a bug in the strategy, not a gap in the data,
                # so it fails loudly.  Silently dropping it would let a strategy
                # whose signal had gone NaN keep "working" at a smaller size.
                raise InvalidOrderError(
                    "strategy produced a non-finite target weight",
                    symbol=symbol,
                    weight=target.weight,
                    decision_ts=str(decision_ts),
                )
            price = marks.get(symbol)
            # ``math.isfinite`` and not just ``price <= 0``: NaN fails every
            # comparison, so ``NaN <= 0`` is False and a NaN mark used to pass
            # this guard.  It then produced a NaN quantity that blew up in
            # ``Order.__post_init__`` with "order quantity must be a positive
            # finite number" -- an error about the symptom, three frames away
            # from the missing price that caused it, which aborted the whole run.
            if price is None or not math.isfinite(price) or price <= 0:
                log.debug(
                    "order.skipped_unusable_mark",
                    symbol=symbol,
                    price=price,
                    ts=str(decision_ts),
                )
                continue
            if (
                abs(target.weight * scale - current_weights.get(symbol, 0.0))
                < cfg.rebalance_tolerance
            ):
                continue

            bar = bars.get(symbol)
            volatility = self._annualised_volatility(market, symbol, decision_ts, cfg)
            sizing = cfg.sizer.target_quantity(
                symbol=symbol,
                target_weight=target.weight * scale,
                price=price,
                equity=equity,
                volatility=volatility,
                bar_volume=bar.volume if bar is not None and bar.volume == bar.volume else None,
            )
            delta = sizing.quantity - portfolio.quantity(symbol)
            if abs(delta) < 1e-9:
                continue
            if abs(delta) * price < cfg.min_trade_notional:
                continue

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            if not self._passes_cash_check(portfolio, side, abs(delta), price, equity, cfg):
                log.debug("order.skipped_cash_limit", symbol=symbol, quantity=delta)
                continue
            orders.append(
                Order(
                    order_id=next_order_id(),
                    symbol=symbol,
                    side=side,
                    quantity=abs(delta),
                    decision_ts=decision_ts,
                    tag=target.reason,
                )
            )
        return orders

    @staticmethod
    def _annualised_volatility(
        market: MarketData, symbol: str, ts: dt.datetime, cfg: EngineConfig
    ) -> float | None:
        """Annualised trailing volatility for the position sizer.

        Inclusive of ``ts``'s bar: the decision is being taken at that bar's
        close, so its return is already known and excluding it would discard
        real information.
        """
        return market.trailing_volatility(
            symbol,
            ts,
            lookback=cfg.volatility_lookback,
            field=cfg.price_field,
            inclusive=True,
            annualise=True,
            minimum=20,
        )

    @staticmethod
    def _passes_cash_check(
        portfolio: Portfolio,
        side: OrderSide,
        quantity: float,
        price: float,
        equity: float,
        cfg: EngineConfig,
    ) -> bool:
        """Reject a buy that would breach the cash floor.

        Approximate: it uses the mark price rather than the (unknown) fill price
        and ignores commission.  The approximation is conservative in the sense
        that costs only push cash further down, so a marginal order that passes
        here can still leave cash slightly below the floor.  The floor is a risk
        control, not an accounting invariant, so a small overshoot is acceptable;
        the accounting identity is checked exactly elsewhere.
        """
        if side is not OrderSide.BUY:
            return True
        floor = -cfg.max_borrow_fraction * equity
        return portfolio.cash - quantity * price >= floor


class _BenchmarkTracker:
    """Buy-and-hold in one symbol, on the same terms a strategy would get.

    Two details matter and both were wrong in an earlier version.

    **Entry timing.**  The benchmark enters on the *second* bar of the run, not
    the first.  The engine's first decision is taken at the close of bar 0 and
    can only fill at bar 1, so a benchmark that buys at bar 0's open captures a
    move no strategy could have captured.  On a long sample that is a rounding
    error; on a sample whose first overnight gap is large it is the whole
    result, and it biases every benchmark-relative metric -- alpha, information
    ratio, "did the strategy beat the market" -- in the benchmark's favour.
    That is look-ahead bias in the benchmark, which is exactly the failure this
    project exists to prevent, pointing the other way.

    **Costs.**  Applied once, at entry.  A benchmark charged daily rebalancing
    costs would be unfairly weak; one charged nothing would be unfairly strong.
    A single entry cost is what an investor buying the ETF actually pays.
    """

    __slots__ = ("_bars_seen", "_entered", "_execution", "cash", "equity", "quantity", "symbol")

    def __init__(self, symbol: str | None, initial_cash: float, execution: ExecutionModel) -> None:
        self.symbol = symbol
        self.cash = initial_cash
        self.quantity = 0.0
        self.equity: float | None = None if symbol is None else initial_cash
        self._entered = False
        self._bars_seen = 0
        self._execution = execution

    def update(self, bars: Mapping[str, Bar], marks: Mapping[str, float]) -> None:
        if self.symbol is None:
            return
        self._bars_seen += 1
        bar = bars.get(self.symbol)
        # ``_bars_seen > 1``: never enter on the first bar of the run.
        if not self._entered and bar is not None and self._bars_seen > 1:
            reference = self._execution.reference_price(bar)
            price = self._execution.costs.slippage.fill_price(
                OrderSide.BUY, reference, self.cash / reference, bar_volume=None
            )
            quantity = self.cash / price
            commission = self._execution.costs.commission.charge(quantity, price)
            quantity = max((self.cash - commission) / price, 0.0)
            self.quantity = quantity
            self.cash -= quantity * price + commission
            self._entered = True
        mark = marks.get(self.symbol)
        if mark is not None:
            self.equity = self.cash + self.quantity * mark


__all__ = ["REBALANCE_FREQUENCIES", "BacktestEngine", "EngineConfig", "EngineOutput"]

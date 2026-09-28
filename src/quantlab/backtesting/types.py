"""Value types for the backtesting engine.

Everything here is immutable or explicitly mutable-with-invariants.  The engine
is easier to reason about when the objects flowing through it cannot be edited
behind your back, and the tests can compare whole objects instead of poking at
attributes.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field

from quantlab.db.enums import OrderSide, OrderStatus, OrderType
from quantlab.exceptions import InvalidOrderError

_order_counter = itertools.count(1)


def next_order_id() -> str:
    return f"ORD-{next(_order_counter):08d}"


@dataclass(frozen=True, slots=True)
class Bar:
    """One session of price data for one symbol."""

    symbol: str
    ts: dt.datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    available_at: dt.datetime

    def price(self, field_name: str) -> float:
        value = getattr(self, field_name, None)
        if value is None:
            raise InvalidOrderError("unknown price field", field=field_name)
        return float(value)


@dataclass(frozen=True, slots=True)
class Order:
    """An intent to trade, stamped with the time the decision was made.

    ``decision_ts`` is set by the engine from the bar that produced the signal
    and is never supplied by the strategy.  A strategy cannot backdate its own
    orders.
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    decision_ts: dt.datetime
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        if not (self.quantity > 0) or self.quantity != self.quantity:  # NaN-safe
            raise InvalidOrderError(
                "order quantity must be a positive finite number",
                symbol=self.symbol,
                quantity=self.quantity,
            )
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise InvalidOrderError("limit order without a limit price", symbol=self.symbol)

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side is OrderSide.BUY else -self.quantity


@dataclass(frozen=True, slots=True)
class Fill:
    """The result of executing an order against a later bar."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    reference_price: float
    fill_price: float
    commission: float
    slippage_cost: float
    decision_ts: dt.datetime
    execution_ts: dt.datetime

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side is OrderSide.BUY else -self.quantity

    @property
    def notional(self) -> float:
        return self.quantity * self.fill_price

    @property
    def cash_delta(self) -> float:
        """Change in cash.  Buys consume cash, sells release it; costs always cost."""
        gross = -self.signed_quantity * self.fill_price
        return gross - self.commission

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost


@dataclass(slots=True)
class Position:
    """A holding.  ``quantity`` is negative for a short.

    ``avg_cost`` uses the weighted-average method and is reset when the position
    flips sign, which is the convention that makes realised P&L on a
    long-to-short reversal come out right.
    """

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realised_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-12

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealised_pnl(self, price: float) -> float:
        return self.quantity * (price - self.avg_cost)

    def apply_fill(self, signed_quantity: float, price: float) -> float:
        """Update the position for a fill.  Returns realised P&L from this fill."""
        old_qty = self.quantity
        new_qty = old_qty + signed_quantity
        realised = 0.0

        same_direction = old_qty == 0 or (old_qty > 0) == (signed_quantity > 0)
        if same_direction:
            total_cost = self.avg_cost * old_qty + price * signed_quantity
            self.avg_cost = total_cost / new_qty if new_qty != 0 else 0.0
        else:
            closing = min(abs(signed_quantity), abs(old_qty))
            direction = 1.0 if old_qty > 0 else -1.0
            realised = closing * (price - self.avg_cost) * direction
            if abs(signed_quantity) > abs(old_qty):
                # Position flipped: the residual opens a new position at ``price``.
                self.avg_cost = price
            elif abs(new_qty) < 1e-12:
                self.avg_cost = 0.0
            # Otherwise a partial close leaves avg_cost unchanged, which is correct.

        self.quantity = 0.0 if abs(new_qty) < 1e-12 else new_qty
        self.realised_pnl += realised
        return realised


@dataclass(frozen=True, slots=True)
class TargetWeight:
    """A strategy's desired portfolio weight for one symbol.

    Weights are fractions of *equity*, signed: ``-0.5`` means a short worth half
    the portfolio.  Using weights rather than share counts keeps strategies
    independent of portfolio size, which is what makes them comparable.
    """

    symbol: str
    weight: float
    reason: str = ""


@dataclass(slots=True)
class OrderRecord:
    """Bookkeeping for an order's lifecycle, used for persistence and audit."""

    order: Order
    status: OrderStatus = OrderStatus.PENDING
    execution_ts: dt.datetime | None = None
    reject_reason: str | None = None
    fill: Fill | None = None


@dataclass(slots=True)
class Snapshot:
    """End-of-bar portfolio state."""

    ts: dt.datetime
    cash: float
    positions_value: float
    equity: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    n_positions: int
    period_return: float
    turnover: float
    benchmark_equity: float | None = None
    holdings: dict[str, float] = field(default_factory=dict)
    #: Per-symbol mark prices used to value ``holdings`` on this bar.  Recorded
    #: alongside holdings so that a persisted position row carries the price the
    #: engine actually valued it at, rather than a price looked up later (which
    #: would be a second, possibly inconsistent, source of truth).  Empty when
    #: ``record_holdings`` is off.
    marks: dict[str, float] = field(default_factory=dict)


__all__ = [
    "Bar",
    "Fill",
    "Order",
    "OrderRecord",
    "Position",
    "Snapshot",
    "TargetWeight",
    "next_order_id",
]

"""Order execution against a future bar.

The temporal rule, stated once and enforced here
------------------------------------------------
An order carries ``decision_ts``, the timestamp of the bar whose information
produced it.  :meth:`ExecutionModel.execute` refuses to fill it against any bar
whose ``ts`` is not strictly later.  That single assertion is what makes the
whole backtest defensible, so it is a hard error rather than a warning, it is
checked on every fill rather than in a review, and there is a test that
deliberately violates it.

Reference price
---------------
``NEXT_OPEN`` fills at the following session's open, which is the realistic
choice for a strategy that computes signals after the close.  ``NEXT_CLOSE``
fills at the following session's close, modelling an execution algorithm that
works the order through the day.  There is no option to fill at the decision
bar's own close: that price is the one that generated the signal, and using it
is the canonical look-ahead bug.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantlab.backtesting.costs import CostModel
from quantlab.backtesting.types import Bar, Fill, Order
from quantlab.db.enums import ExecutionTiming, OrderSide, OrderType
from quantlab.exceptions import LookAheadError
from quantlab.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Rejection:
    order: Order
    reason: str


@dataclass(frozen=True, slots=True)
class ExecutionModel:
    """Turns an order plus a later bar into a fill (or a rejection)."""

    timing: ExecutionTiming = ExecutionTiming.NEXT_OPEN
    costs: CostModel = CostModel()
    #: Reject an order larger than this fraction of the execution bar's volume.
    #: A backtest that assumes it can trade half a day's volume is fiction.
    max_participation: float | None = 0.25
    #: Estimate of the bar's return volatility, used by impact-aware slippage.
    default_volatility: float = 0.02

    def reference_price(self, bar: Bar) -> float:
        return bar.open if self.timing is ExecutionTiming.NEXT_OPEN else bar.close

    def execute(
        self,
        order: Order,
        bar: Bar,
        *,
        volatility: float | None = None,
    ) -> Fill | Rejection:
        # --- the temporal guard ------------------------------------------
        if bar.ts <= order.decision_ts:
            raise LookAheadError(
                "execution bar is not strictly after the decision bar",
                symbol=order.symbol,
                decision_ts=order.decision_ts.isoformat(),
                execution_ts=bar.ts.isoformat(),
            )
        if bar.symbol != order.symbol:
            raise LookAheadError(
                "execution bar is for a different symbol",
                order_symbol=order.symbol,
                bar_symbol=bar.symbol,
            )

        reference = self.reference_price(bar)
        if not (reference > 0):
            return Rejection(order, "non_positive_reference_price")

        # ``bar.volume == bar.volume`` is a NaN check: volume is optional and a
        # missing value must not silently reject the order.
        has_volume = bar.volume == bar.volume and bar.volume > 0
        if (
            self.max_participation is not None
            and has_volume
            and order.quantity > self.max_participation * bar.volume
        ):
            return Rejection(order, "exceeds_max_participation")

        if order.order_type is OrderType.LIMIT:
            assert order.limit_price is not None
            reachable = (
                bar.low <= order.limit_price
                if order.side is OrderSide.BUY
                else bar.high >= order.limit_price
            )
            if not reachable:
                return Rejection(order, "limit_not_reached")
            # Fill at the better of the limit and the reference price, which is
            # the conservative assumption for a marketable limit order.
            reference = (
                min(reference, order.limit_price)
                if order.side is OrderSide.BUY
                else max(reference, order.limit_price)
            )

        fill_price = self.costs.slippage.fill_price(
            order.side,
            reference,
            order.quantity,
            bar_volume=None if bar.volume != bar.volume else bar.volume,
            volatility=volatility if volatility is not None else self.default_volatility,
        )
        commission = self.costs.commission.charge(order.quantity, fill_price)
        slippage_cost = abs(fill_price - reference) * order.quantity

        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            reference_price=reference,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slippage_cost,
            decision_ts=order.decision_ts,
            execution_ts=bar.ts,
        )

    def describe(self) -> dict[str, object]:
        return {
            "timing": str(self.timing),
            "max_participation": self.max_participation,
            "costs": self.costs.describe(),
        }


__all__ = ["ExecutionModel", "Rejection"]

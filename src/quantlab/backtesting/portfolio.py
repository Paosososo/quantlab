"""Portfolio state and accounting.

The accounting identity ``equity = cash + sum(quantity * price)`` is checked
after every bar.  That check has caught more bugs during development of this
project than any unit test: a mis-signed short, a commission charged twice, a
fill applied without a cash movement all break it immediately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from quantlab.backtesting.types import Fill, Position
from quantlab.exceptions import BacktestError
from quantlab.logging import get_logger

log = get_logger(__name__)

#: Relative tolerance for the accounting identity.  1e-9 of equity is far below
#: any economically meaningful amount but far above float64 round-off on a
#: 10,000-trade run.
ACCOUNTING_TOLERANCE = 1e-9


@dataclass(slots=True)
class Portfolio:
    initial_cash: float
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    realised_pnl: float = 0.0
    #: Running sum of |traded notional|, the numerator of turnover.
    traded_notional: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise BacktestError("initial cash must be positive", value=self.initial_cash)
        self.cash = float(self.initial_cash)

    # -- queries -----------------------------------------------------------
    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol=symbol))

    def quantity(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return 0.0 if pos is None else pos.quantity

    def positions_value(self, prices: Mapping[str, float]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue
            price = prices.get(symbol)
            if price is None:
                raise BacktestError("no price to mark a held position", symbol=symbol)
            total += pos.market_value(price)
        return total

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + self.positions_value(prices)

    def gross_exposure(self, prices: Mapping[str, float]) -> float:
        return sum(
            abs(pos.market_value(prices[pos.symbol]))
            for pos in self.positions.values()
            if not pos.is_flat and pos.symbol in prices
        )

    def net_exposure(self, prices: Mapping[str, float]) -> float:
        return sum(
            pos.market_value(prices[pos.symbol])
            for pos in self.positions.values()
            if not pos.is_flat and pos.symbol in prices
        )

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        equity = self.equity(prices)
        if abs(equity) < 1e-12:
            return {}
        return {
            pos.symbol: pos.market_value(prices[pos.symbol]) / equity
            for pos in self.positions.values()
            if not pos.is_flat and pos.symbol in prices
        }

    def holdings(self) -> dict[str, float]:
        return {s: p.quantity for s, p in self.positions.items() if not p.is_flat}

    @property
    def n_positions(self) -> int:
        return sum(1 for p in self.positions.values() if not p.is_flat)

    # -- mutation ----------------------------------------------------------
    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill: move cash, update the position, accrue costs."""
        position = self.position(fill.symbol)
        realised = position.apply_fill(fill.signed_quantity, fill.fill_price)
        self.cash += fill.cash_delta
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage_cost
        self.realised_pnl += realised
        self.traded_notional += abs(fill.notional)
        return realised

    # -- invariants --------------------------------------------------------
    def assert_accounting_identity(self, prices: Mapping[str, float], *, label: str = "") -> float:
        """Verify ``equity == cash + positions_value`` and return equity.

        This is trivially true by construction of :meth:`equity`, so the check
        that matters is the stronger one below: equity must also equal the
        initial cash plus every cash flow and mark-to-market move that has
        happened.  Any divergence means a fill was applied to a position without
        a matching cash movement.
        """
        positions_value = self.positions_value(prices)
        equity = self.cash + positions_value
        unrealised = sum(
            pos.unrealised_pnl(prices[pos.symbol])
            for pos in self.positions.values()
            if not pos.is_flat and pos.symbol in prices
        )
        expected = self.initial_cash + self.realised_pnl + unrealised - self.total_commission
        scale = max(abs(equity), abs(self.initial_cash), 1.0)
        if abs(equity - expected) > ACCOUNTING_TOLERANCE * scale:
            raise BacktestError(
                "portfolio accounting identity violated",
                label=label,
                equity=equity,
                expected=expected,
                difference=equity - expected,
                cash=self.cash,
                positions_value=positions_value,
            )
        return equity


__all__ = ["ACCOUNTING_TOLERANCE", "Portfolio"]

"""Portfolio accounting.

Position averaging and the cash identity are the two places where a bug produces
a plausible-looking equity curve rather than a crash, so they are tested
exhaustively including the awkward cases: shorts, partial closes, and a position
that flips from long to short in a single fill.
"""

from __future__ import annotations

import datetime as dt

import pytest

from quantlab.backtesting.portfolio import Portfolio
from quantlab.backtesting.types import Fill, Position
from quantlab.db.enums import OrderSide
from quantlab.exceptions import BacktestError
from quantlab.timeutils import UTC

pytestmark = pytest.mark.unit


def make_fill(
    symbol: str, side: OrderSide, quantity: float, price: float, commission: float = 0.0
) -> Fill:
    return Fill(
        order_id="X",
        symbol=symbol,
        side=side,
        quantity=quantity,
        reference_price=price,
        fill_price=price,
        commission=commission,
        slippage_cost=0.0,
        decision_ts=dt.datetime(2020, 1, 1, tzinfo=UTC),
        execution_ts=dt.datetime(2020, 1, 2, tzinfo=UTC),
    )


class TestPosition:
    def test_opening_a_long_sets_average_cost(self):
        pos = Position("A")
        assert pos.apply_fill(100, 10.0) == 0.0
        assert pos.quantity == 100
        assert pos.avg_cost == pytest.approx(10.0)

    def test_adding_to_a_long_averages_the_cost(self):
        pos = Position("A")
        pos.apply_fill(100, 10.0)
        pos.apply_fill(100, 20.0)
        assert pos.quantity == 200
        assert pos.avg_cost == pytest.approx(15.0)

    def test_closing_realises_profit(self):
        pos = Position("A")
        pos.apply_fill(100, 10.0)
        realised = pos.apply_fill(-100, 12.0)
        assert realised == pytest.approx(200.0)
        assert pos.is_flat
        assert pos.avg_cost == 0.0

    def test_partial_close_leaves_average_cost_unchanged(self):
        pos = Position("A")
        pos.apply_fill(100, 10.0)
        realised = pos.apply_fill(-40, 15.0)
        assert realised == pytest.approx(200.0)
        assert pos.quantity == 60
        assert pos.avg_cost == pytest.approx(10.0)

    def test_short_profits_when_price_falls(self):
        pos = Position("A")
        pos.apply_fill(-100, 20.0)
        assert pos.quantity == -100
        assert pos.avg_cost == pytest.approx(20.0)
        realised = pos.apply_fill(100, 15.0)
        assert realised == pytest.approx(500.0)

    def test_flip_from_long_to_short_realises_then_reopens(self):
        """Sell 150 against a 100 long: realise on 100, open a 50 short at the fill price."""
        pos = Position("A")
        pos.apply_fill(100, 10.0)
        realised = pos.apply_fill(-150, 12.0)
        assert realised == pytest.approx(200.0)
        assert pos.quantity == -50
        assert pos.avg_cost == pytest.approx(12.0)

    def test_unrealised_pnl_uses_the_mark(self):
        pos = Position("A")
        pos.apply_fill(100, 10.0)
        assert pos.unrealised_pnl(11.0) == pytest.approx(100.0)
        assert pos.market_value(11.0) == pytest.approx(1100.0)


class TestPortfolio:
    def test_initial_cash_must_be_positive(self):
        with pytest.raises(BacktestError):
            Portfolio(initial_cash=0.0)

    def test_a_buy_moves_cash_by_notional_plus_commission(self):
        pf = Portfolio(initial_cash=10_000.0)
        pf.apply_fill(make_fill("A", OrderSide.BUY, 100, 50.0, commission=7.0))
        assert pf.cash == pytest.approx(10_000.0 - 5_000.0 - 7.0)
        assert pf.quantity("A") == 100

    def test_a_sell_returns_cash_minus_commission(self):
        pf = Portfolio(initial_cash=10_000.0)
        pf.apply_fill(make_fill("A", OrderSide.BUY, 100, 50.0))
        pf.apply_fill(make_fill("A", OrderSide.SELL, 100, 55.0, commission=3.0))
        assert pf.cash == pytest.approx(10_000.0 + 500.0 - 3.0)
        assert pf.quantity("A") == 0

    def test_shorting_credits_cash(self):
        pf = Portfolio(initial_cash=10_000.0)
        pf.apply_fill(make_fill("A", OrderSide.SELL, 100, 50.0))
        assert pf.cash == pytest.approx(15_000.0)
        assert pf.equity({"A": 50.0}) == pytest.approx(10_000.0)

    def test_equity_is_cash_plus_marked_positions(self):
        pf = Portfolio(initial_cash=10_000.0)
        pf.apply_fill(make_fill("A", OrderSide.BUY, 100, 50.0))
        assert pf.equity({"A": 60.0}) == pytest.approx(5_000.0 + 6_000.0)

    def test_accounting_identity_holds_through_a_round_trip(self):
        pf = Portfolio(initial_cash=100_000.0)
        pf.apply_fill(make_fill("A", OrderSide.BUY, 500, 100.0, commission=2.5))
        pf.assert_accounting_identity({"A": 100.0})
        pf.apply_fill(make_fill("A", OrderSide.BUY, 300, 110.0, commission=1.5))
        pf.assert_accounting_identity({"A": 105.0})
        pf.apply_fill(make_fill("A", OrderSide.SELL, 800, 120.0, commission=4.0))
        equity = pf.assert_accounting_identity({"A": 120.0})
        assert equity == pytest.approx(pf.cash)

    def test_accounting_identity_detects_a_desynchronised_position(self):
        """Move a position without a matching cash flow; the identity must catch it."""
        pf = Portfolio(initial_cash=100_000.0)
        pf.apply_fill(make_fill("A", OrderSide.BUY, 100, 50.0))
        pf.position("A").quantity += 50  # simulate the bug
        with pytest.raises(BacktestError, match="accounting identity"):
            pf.assert_accounting_identity({"A": 50.0})

    def test_marking_a_held_position_without_a_price_raises(self):
        pf = Portfolio(initial_cash=10_000.0)
        pf.apply_fill(make_fill("A", OrderSide.BUY, 10, 50.0))
        with pytest.raises(BacktestError, match="no price"):
            pf.equity({})

    def test_weights_sum_to_gross_exposure_over_equity(self):
        pf = Portfolio(initial_cash=10_000.0)
        pf.apply_fill(make_fill("A", OrderSide.BUY, 50, 100.0))
        pf.apply_fill(make_fill("B", OrderSide.SELL, 20, 100.0))
        marks = {"A": 100.0, "B": 100.0}
        weights = pf.weights(marks)
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(-0.2)
        assert pf.gross_exposure(marks) == pytest.approx(7_000.0)
        assert pf.net_exposure(marks) == pytest.approx(3_000.0)

    def test_costs_accumulate(self):
        pf = Portfolio(initial_cash=10_000.0)
        pf.apply_fill(make_fill("A", OrderSide.BUY, 10, 10.0, commission=1.0))
        pf.apply_fill(make_fill("A", OrderSide.SELL, 10, 10.0, commission=2.0))
        assert pf.total_commission == pytest.approx(3.0)
        assert pf.traded_notional == pytest.approx(200.0)

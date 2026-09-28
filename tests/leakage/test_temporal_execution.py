"""The project's central claim, tested directly.

A signal computed from information at bar ``t`` must not be executed at a price
from bar ``t`` or earlier.  These tests try to break that in several ways and
assert the system refuses.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pandas as pd
import pytest

from quantlab.backtesting.costs import ZERO_COST
from quantlab.backtesting.engine import BacktestEngine, EngineConfig
from quantlab.backtesting.execution import ExecutionModel
from quantlab.backtesting.market import MarketData
from quantlab.backtesting.sizing import TargetWeightSizer
from quantlab.backtesting.strategy import DecisionContext, Strategy
from quantlab.backtesting.types import Bar, Order, TargetWeight
from quantlab.db.enums import ExecutionTiming, OrderSide
from quantlab.exceptions import BacktestError, LookAheadError
from quantlab.timeutils import UTC
from tests.conftest import make_bars

pytestmark = [pytest.mark.leakage, pytest.mark.unit]

FRACTIONAL = TargetWeightSizer(allow_fractional=True, max_participation=None)


def zero_cost_engine(**overrides) -> BacktestEngine:
    defaults = {
        "initial_cash": 1000.0,
        "execution": ExecutionModel(costs=ZERO_COST, max_participation=None),
        "sizer": FRACTIONAL,
        "max_borrow_fraction": 1.0,
    }
    defaults.update(overrides)
    return BacktestEngine(EngineConfig(**defaults))


class LongFromBar(Strategy):
    """Go long once the decision bar index reaches ``trigger``."""

    name = "long_from_bar"

    def __init__(self, symbol: str, trigger: int) -> None:
        self.symbol = symbol
        self.trigger = trigger
        self._seen = -1

    def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:
        self._seen += 1
        return [TargetWeight(self.symbol, 1.0 if self._seen >= self.trigger else 0.0)]


class TestExecutionModelGuard:
    def test_filling_on_the_decision_bar_raises(self):
        """The bug this whole project is about, attempted directly."""
        ts = dt.datetime(2020, 1, 2, tzinfo=UTC)
        order = Order("O1", "A", OrderSide.BUY, 10, decision_ts=ts)
        bar = Bar("A", ts, 100.0, 101.0, 99.0, 100.0, 100.0, 1e6, ts)
        with pytest.raises(LookAheadError, match="strictly after"):
            ExecutionModel().execute(order, bar)

    def test_filling_on_an_earlier_bar_raises(self):
        order = Order("O1", "A", OrderSide.BUY, 10, decision_ts=dt.datetime(2020, 1, 3, tzinfo=UTC))
        earlier = dt.datetime(2020, 1, 2, tzinfo=UTC)
        bar = Bar("A", earlier, 100.0, 101.0, 99.0, 100.0, 100.0, 1e6, earlier)
        with pytest.raises(LookAheadError):
            ExecutionModel().execute(order, bar)

    def test_filling_the_next_bar_is_allowed(self):
        decision = dt.datetime(2020, 1, 2, tzinfo=UTC)
        execution = dt.datetime(2020, 1, 3, tzinfo=UTC)
        order = Order("O1", "A", OrderSide.BUY, 10, decision_ts=decision)
        bar = Bar("A", execution, 100.0, 101.0, 99.0, 100.5, 100.5, 1e6, execution)
        fill = ExecutionModel(costs=ZERO_COST).execute(order, bar)
        assert fill.execution_ts == execution
        assert fill.reference_price == 100.0  # next open, not the decision bar's close

    def test_a_bar_for_another_symbol_is_refused(self):
        decision = dt.datetime(2020, 1, 2, tzinfo=UTC)
        execution = dt.datetime(2020, 1, 3, tzinfo=UTC)
        order = Order("O1", "A", OrderSide.BUY, 10, decision_ts=decision)
        bar = Bar("B", execution, 100.0, 101.0, 99.0, 100.0, 100.0, 1e6, execution)
        with pytest.raises(LookAheadError, match="different symbol"):
            ExecutionModel().execute(order, bar)


class TestEngineOrdering:
    def test_every_fill_is_strictly_after_its_decision(self, two_symbol_frame):
        from quantlab.backtesting.strategy import MovingAverageCrossover

        market = MarketData(two_symbol_frame)
        out = zero_cost_engine().run(
            MovingAverageCrossover(["SYN_A", "SYN_B"], fast=10, slow=30), market
        )
        assert out.fills
        assert all(f.execution_ts > f.decision_ts for f in out.fills)

    def test_the_overnight_gap_is_not_captured(self):
        """The sharpest version of the test.

        Prices are flat at 100 through bar 3, then gap to 200 *overnight* into
        bar 4.  A strategy that decides to buy using bar 3's close fills at bar
        4's open of 200, so it earns nothing from the gap.  An engine that
        filled at bar 3's close of 100 would double its money instantly.
        """
        closes = [100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
        opens = [100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0]
        market = MarketData(make_bars("A", closes=closes, opens=opens))
        out = zero_cost_engine().run(LongFromBar("A", trigger=3), market)

        final_equity = out.snapshots[-1].equity
        assert final_equity == pytest.approx(1000.0), (
            f"the strategy bought after the gap and must not profit from it; got {final_equity}"
        )
        first_fill = out.fills[0]
        assert first_fill.reference_price == pytest.approx(200.0)

    def test_an_earlier_decision_does_capture_the_gap(self):
        """The complement of the previous test.

        Deciding at bar 2 fills at bar 3's open of 100, before the gap, so the
        gain *is* earned.  Without this, the previous test would also pass on an
        engine that simply never trades.
        """
        closes = [100.0, 100.0, 100.0, 100.0, 200.0, 200.0]
        opens = [100.0, 100.0, 100.0, 100.0, 200.0, 200.0]
        market = MarketData(make_bars("A", closes=closes, opens=opens))
        out = zero_cost_engine().run(LongFromBar("A", trigger=2), market)
        assert out.fills[0].reference_price == pytest.approx(100.0)
        assert out.snapshots[-1].equity > 1900.0

    def test_next_close_timing_also_respects_ordering(self):
        closes = [100.0, 100.0, 100.0, 100.0, 200.0, 200.0]
        market = MarketData(make_bars("A", closes=closes, opens=closes))
        out = zero_cost_engine(
            execution=ExecutionModel(
                timing=ExecutionTiming.NEXT_CLOSE, costs=ZERO_COST, max_participation=None
            )
        ).run(LongFromBar("A", trigger=3), market)
        assert all(f.execution_ts > f.decision_ts for f in out.fills)
        assert out.fills[0].reference_price == pytest.approx(200.0)


class TestDecisionContextIsolation:
    def test_the_strategy_cannot_see_future_bars(self):
        seen: list[tuple[pd.Timestamp, int]] = []

        class Recorder(Strategy):
            name = "recorder"

            def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:
                view = ctx.prices["A"]
                seen.append((pd.Timestamp(ctx.ts), len(view)))
                # The most recent visible close must be this bar's own close.
                assert view.latest("close") is not None
                return []

        closes = [100.0 + i for i in range(20)]
        market = MarketData(make_bars("A", closes=closes))
        zero_cost_engine().run(Recorder(), market)

        # On bar i the strategy sees exactly i+1 bars: everything up to and
        # including today, and nothing after.
        for index, (_, visible) in enumerate(seen):
            assert visible == index + 1

    def test_reaching_past_the_as_of_instant_raises(self):
        class Cheater(Strategy):
            name = "cheater"

            def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:
                view = ctx.prices["A"]
                view.value_at(ctx.as_of + dt.timedelta(days=5), "close")
                return []

        market = MarketData(make_bars("A", closes=[100.0] * 10))
        with pytest.raises(LookAheadError):
            zero_cost_engine().run(Cheater(), market)


class TestPredictionAvailability:
    def _market(self) -> MarketData:
        return MarketData(make_bars("A", closes=[100.0] * 10))

    def test_predictions_without_availability_are_refused(self):
        from quantlab.backtesting.strategy import PredictionStrategy

        market = self._market()
        predictions = pd.DataFrame(
            {
                "symbol": ["A"] * 3,
                "ts": market.timeline[:3],
                "y_pred": [0.01, 0.02, 0.03],
            }
        )
        with pytest.raises(BacktestError, match="available_at"):
            zero_cost_engine().run(PredictionStrategy(["A"]), market, predictions=predictions)

    def test_a_prediction_published_later_is_invisible(self):
        """A forecast stamped with a future availability must not be actionable."""
        from quantlab.backtesting.strategy import PredictionStrategy

        market = self._market()
        timeline = market.timeline
        predictions = pd.DataFrame(
            {
                "symbol": ["A"] * len(timeline),
                "ts": timeline,
                # Published a year late: nothing should ever act on these.
                "available_at": timeline + pd.Timedelta(days=365),
                "y_pred": [0.05] * len(timeline),
            }
        )
        out = zero_cost_engine().run(
            PredictionStrategy(["A"], long_threshold=0.0), market, predictions=predictions
        )
        assert not out.fills

    def test_a_prediction_available_at_the_close_is_actionable(self):
        from quantlab.backtesting.strategy import PredictionStrategy

        market = self._market()
        tidy = market.tidy
        predictions = pd.DataFrame(
            {
                "symbol": tidy["symbol"],
                "ts": tidy["ts"],
                "available_at": tidy["available_at"],
                "y_pred": 0.05,
            }
        )
        out = zero_cost_engine().run(
            PredictionStrategy(["A"], long_threshold=0.0), market, predictions=predictions
        )
        assert out.fills
        assert all(f.execution_ts > f.decision_ts for f in out.fills)

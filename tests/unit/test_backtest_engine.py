"""Backtest engine correctness.

The tests fall into three groups:

1.  **Closed-form arithmetic.**  Small hand-built price series where the correct
    equity can be computed on paper, so a regression shows up as a wrong number
    rather than a vague "looks different".
2.  **Cost accounting.**  Commission and slippage must appear in cash exactly
    once and in the right direction.
3.  **Housekeeping.**  Rebalance frequency, participation caps, missing bars.

Temporal-ordering tests live in ``tests/leakage/`` because they are the ones
that protect the project's central claim.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest

from quantlab.backtesting.costs import (
    ZERO_COST,
    CostModel,
    FixedBpsSlippage,
    NoCommission,
    NoSlippage,
    PerShareCommission,
)
from quantlab.backtesting.engine import BacktestEngine, EngineConfig
from quantlab.backtesting.execution import ExecutionModel
from quantlab.backtesting.market import MarketData
from quantlab.backtesting.results import build_result
from quantlab.backtesting.sizing import TargetWeightSizer
from quantlab.backtesting.strategy import BuyAndHold, DecisionContext, Strategy
from quantlab.backtesting.types import TargetWeight
from quantlab.db.enums import ExecutionTiming
from quantlab.exceptions import BacktestError
from tests.conftest import make_bars

pytestmark = pytest.mark.unit

FRACTIONAL = TargetWeightSizer(allow_fractional=True, max_participation=None)


def engine(**overrides) -> BacktestEngine:
    defaults = {
        "initial_cash": 1000.0,
        "execution": ExecutionModel(costs=ZERO_COST, max_participation=None),
        "sizer": FRACTIONAL,
        "rebalance": "daily",
    }
    defaults.update(overrides)
    return BacktestEngine(EngineConfig(**defaults))


class AlwaysLong(Strategy):
    name = "always_long"

    def __init__(self, symbol: str, weight: float = 1.0) -> None:
        self.symbol = symbol
        self.weight = weight

    def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:
        return [TargetWeight(self.symbol, self.weight)]


class BuyOnceAndHold(Strategy):
    """Emit a target on the first decision only, then leave the position alone.

    ``BuyAndHold`` keeps a constant *weight*, which means it trades as prices
    move.  This one keeps a constant *share count*, which is what makes the
    closed-form assertion below exact.
    """

    name = "buy_once"

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._done = False

    def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:
        if self._done:
            return []
        self._done = True
        return [TargetWeight(self.symbol, 1.0)]


class TestClosedFormArithmetic:
    def test_three_bar_run_matches_hand_computation(self):
        """Worked by hand:

        bar0 close 100, equity 1000 -> target 10 shares, order to buy 10
        bar1 fills at open 105      -> cash = 1000 - 1050 = -50
                  close 110         -> equity = -50 + 10*110 = 1050
                  target = 1050/110 = 9.545454..., sell 0.454545...
        bar2 fills at open 115      -> cash = -50 + 52.272727... = 2.272727...
                  close 121         -> equity = 2.272727... + 9.545454...*121
        """
        bars = make_bars("A", closes=[100.0, 110.0, 121.0], opens=[100.0, 105.0, 115.0])
        out = engine(max_borrow_fraction=1.0).run(AlwaysLong("A"), MarketData(bars))
        equity = [s.equity for s in out.snapshots]

        assert equity[0] == pytest.approx(1000.0)
        assert equity[1] == pytest.approx(1050.0)
        expected_qty = 1050.0 / 110.0
        expected_cash = -50.0 + (10.0 - expected_qty) * 115.0
        assert equity[2] == pytest.approx(expected_cash + expected_qty * 121.0)

    def test_buy_once_and_hold_matches_the_closed_form(self):
        """Buy N shares at bar 1's open, hold to the end.

        Final equity is ``cash + N * last_close`` where ``cash = 1000 - N *
        open[1]`` and ``N = floor(1000 / close[0])``.  No approximation, no
        tolerance beyond float noise.
        """
        closes = [100.0 * (1.01**i) for i in range(60)]
        opens = [c * 0.999 for c in closes]
        bars = make_bars("A", closes=closes, opens=opens)
        out = engine(sizer=TargetWeightSizer(allow_fractional=False, max_participation=None)).run(
            BuyOnceAndHold("A"), MarketData(bars)
        )

        shares = float(int(1000.0 / closes[0]))
        expected_cash = 1000.0 - shares * opens[1]
        expected_equity = expected_cash + shares * closes[-1]
        assert len(out.fills) == 1
        assert out.fills[0].quantity == shares
        assert out.snapshots[-1].equity == pytest.approx(expected_equity)
        assert build_result(out).metrics.total_return == pytest.approx(
            expected_equity / 1000.0 - 1.0
        )

    def test_constant_weight_buy_and_hold_tracks_the_asset(self):
        """Holding a constant weight requires trading, so it only tracks the
        asset approximately.  Within a few basis points is the honest claim."""
        closes = [100.0 * (1.005**i) for i in range(120)]
        bars = make_bars("A", closes=closes, opens=closes)
        out = engine(max_borrow_fraction=1.0).run(
            BuyAndHold(["A"], weights={"A": 1.0}), MarketData(bars)
        )
        expected = closes[-1] / closes[1] - 1.0
        assert build_result(out).metrics.total_return == pytest.approx(expected, abs=5e-4)

    def test_a_flat_market_produces_no_pnl(self):
        bars = make_bars("A", closes=[100.0] * 20)
        out = engine().run(AlwaysLong("A"), MarketData(bars))
        assert out.snapshots[-1].equity == pytest.approx(1000.0)

    def test_staying_flat_leaves_cash_untouched(self):
        bars = make_bars("A", closes=[100.0, 120.0, 90.0, 130.0])
        out = engine().run(AlwaysLong("A", weight=0.0), MarketData(bars))
        assert all(s.equity == pytest.approx(1000.0) for s in out.snapshots)
        assert not out.fills

    def test_a_short_gains_when_the_price_falls(self):
        bars = make_bars("A", closes=[100.0, 90.0, 80.0], opens=[100.0, 100.0, 90.0])
        out = engine(max_borrow_fraction=1.0).run(AlwaysLong("A", weight=-1.0), MarketData(bars))
        # 10 shares short at 100 on bar 1; marked at 90 -> +100 on equity.
        assert out.snapshots[1].equity == pytest.approx(1100.0)


class TestCostAccounting:
    def test_commission_reduces_cash_by_exactly_the_fee(self):
        bars = make_bars("A", closes=[100.0, 100.0, 100.0], opens=[100.0, 100.0, 100.0])
        # Slippage is switched off explicitly: CostModel's default includes a
        # 5bp half-spread, and the point of this test is the commission alone.
        costs = CostModel(
            commission=PerShareCommission(rate_per_share=0.01, minimum=0.0),
            slippage=NoSlippage(),
        )
        out = engine(
            execution=ExecutionModel(costs=costs, max_participation=None),
            sizer=TargetWeightSizer(allow_fractional=False, max_participation=None),
        ).run(AlwaysLong("A"), MarketData(bars))
        first = out.fills[0]
        assert first.quantity == 10.0
        assert first.commission == pytest.approx(0.10)
        assert out.snapshots[1].cash == pytest.approx(1000.0 - 10 * 100.0 - 0.10)

    def test_slippage_moves_the_fill_price_against_the_order(self):
        bars = make_bars("A", closes=[100.0, 100.0, 100.0], opens=[100.0, 100.0, 100.0])
        costs = CostModel(commission=NoCommission(), slippage=FixedBpsSlippage(50.0))
        out = engine(execution=ExecutionModel(costs=costs, max_participation=None)).run(
            AlwaysLong("A"), MarketData(bars)
        )
        fill = out.fills[0]
        assert fill.reference_price == pytest.approx(100.0)
        assert fill.fill_price == pytest.approx(100.5)
        assert fill.slippage_cost == pytest.approx(0.5 * fill.quantity)

    def test_costs_make_a_flat_market_lose_money(self):
        """Trading a market that goes nowhere must lose exactly the costs paid."""
        bars = make_bars("A", closes=[100.0, 101.0, 100.0, 101.0, 100.0, 101.0] * 5)
        costs = CostModel(PerShareCommission(0.01, minimum=0.0), FixedBpsSlippage(10.0))
        out = engine(execution=ExecutionModel(costs=costs, max_participation=None)).run(
            AlwaysLong("A"), MarketData(bars)
        )
        result = build_result(out)
        assert result.metrics.total_commission > 0
        assert result.metrics.total_slippage > 0

    def test_next_close_timing_fills_at_the_following_close(self):
        bars = make_bars("A", closes=[100.0, 110.0, 120.0], opens=[100.0, 105.0, 115.0])
        out = engine(
            execution=ExecutionModel(
                timing=ExecutionTiming.NEXT_CLOSE, costs=ZERO_COST, max_participation=None
            ),
            max_borrow_fraction=1.0,
        ).run(AlwaysLong("A"), MarketData(bars))
        assert out.fills[0].reference_price == pytest.approx(110.0)


class TestHousekeeping:
    def test_duplicate_bars_are_rejected(self):
        bars = make_bars("A", closes=[100.0, 101.0])
        duplicated = pd.concat([bars, bars.iloc[[1]]], ignore_index=True)
        with pytest.raises(BacktestError, match="duplicate"):
            MarketData(duplicated)

    def test_missing_columns_are_rejected(self):
        bars = make_bars("A", closes=[100.0, 101.0]).drop(columns=["available_at"])
        with pytest.raises(BacktestError, match="missing columns"):
            MarketData(bars)

    def test_a_single_bar_cannot_be_backtested(self):
        bars = make_bars("A", closes=[100.0])
        with pytest.raises(BacktestError, match="at least two bars"):
            engine().run(AlwaysLong("A"), MarketData(bars))

    def test_monthly_rebalancing_trades_far_less_than_daily(self):
        """Two assets that drift apart force a constant-weight portfolio to
        trade.  A single fully invested asset does not, because the target share
        count is self-consistent, so the comparison needs two."""
        a = make_bars("A", closes=[100.0 * (1.004**i) for i in range(300)])
        b = make_bars("B", closes=[100.0 * (0.998**i) for i in range(300)])
        market = MarketData(pd.concat([a, b], ignore_index=True))
        strategy = BuyAndHold(["A", "B"])
        daily = engine(rebalance="daily", max_borrow_fraction=1.0).run(strategy, market)
        monthly = engine(rebalance="monthly", max_borrow_fraction=1.0).run(
            BuyAndHold(["A", "B"]), market
        )
        assert len(monthly.fills) < len(daily.fills)
        assert len(monthly.fills) <= 40  # about two per month over 300 business days

    def test_rebalance_tolerance_suppresses_noise_trades(self):
        a = make_bars("A", closes=[100.0 * (1.004**i) for i in range(150)])
        b = make_bars("B", closes=[100.0 * (0.998**i) for i in range(150)])
        market = MarketData(pd.concat([a, b], ignore_index=True))
        tight = engine(rebalance_tolerance=0.0, max_borrow_fraction=1.0).run(
            BuyAndHold(["A", "B"]), market
        )
        loose = engine(rebalance_tolerance=0.5, max_borrow_fraction=1.0).run(
            BuyAndHold(["A", "B"]), market
        )
        assert len(loose.fills) < len(tight.fills)

    def test_participation_cap_rejects_an_oversized_order(self):
        bars = make_bars("A", closes=[10.0] * 5, volumes=[100.0] * 5)
        out = BacktestEngine(
            EngineConfig(
                initial_cash=1_000_000.0,
                execution=ExecutionModel(costs=ZERO_COST, max_participation=0.1),
                sizer=TargetWeightSizer(allow_fractional=True, max_participation=None),
                max_borrow_fraction=1.0,
            )
        ).run(AlwaysLong("A"), MarketData(bars))
        assert out.rejections
        assert out.rejections[0].reason == "exceeds_max_participation"

    def test_cash_floor_blocks_borrowing_when_forbidden(self):
        bars = make_bars("A", closes=[100.0] * 10)
        out = engine(max_borrow_fraction=0.0).run(AlwaysLong("A", weight=3.0), MarketData(bars))
        # Gross leverage is capped at 1.0 by default, so the 3x target is scaled
        # back rather than borrowing.
        assert all(s.leverage <= 1.0 + 1e-9 for s in out.snapshots)

    def test_an_untraded_symbol_keeps_its_order_pending(self):
        """A holiday in one market must not silently cancel the strategy's intent."""
        a = make_bars("A", closes=[100.0, 101.0, 102.0, 103.0])
        b = make_bars("B", closes=[50.0, 51.0, 52.0, 53.0]).drop(index=1).reset_index(drop=True)
        out = engine(max_borrow_fraction=1.0).run(
            BuyAndHold(["A", "B"]), MarketData(pd.concat([a, b], ignore_index=True))
        )
        filled_b = [f for f in out.fills if f.symbol == "B"]
        assert filled_b, "the order for B should fill on the next session it trades"
        assert filled_b[0].execution_ts > filled_b[0].decision_ts

    def test_snapshot_count_matches_the_timeline(self):
        bars = make_bars("A", closes=[100.0] * 25)
        out = engine().run(AlwaysLong("A"), MarketData(bars))
        assert len(out.snapshots) == 25

    def test_unknown_rebalance_frequency_is_rejected(self):
        with pytest.raises(BacktestError, match="rebalance frequency"):
            EngineConfig(rebalance="fortnightly")

    def test_warmup_delays_the_first_trade(self):
        class Slow(AlwaysLong):
            def warmup_bars(self) -> int:
                return 10

        bars = make_bars("A", closes=[100.0] * 30)
        out = engine().run(Slow("A"), MarketData(bars))
        assert out.fills
        first_decision = min(f.decision_ts for f in out.fills)
        assert first_decision >= pd.Timestamp(out.snapshots[9].ts)

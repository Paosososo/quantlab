"""Transaction costs.

The sign convention gets its own test class because getting it backwards turns
a cost into a source of alpha, which is a silent and very flattering bug.
"""

from __future__ import annotations

import pytest

from quantlab.backtesting.costs import (
    BPS,
    CostModel,
    FixedBpsSlippage,
    NoCommission,
    NoSlippage,
    PercentCommission,
    PerShareCommission,
    SquareRootImpactSlippage,
)
from quantlab.db.enums import OrderSide

pytestmark = pytest.mark.unit


class TestCommission:
    def test_no_commission_is_free(self):
        assert NoCommission().charge(1000, 50.0) == 0.0

    def test_per_share_uses_the_rate_above_the_minimum(self):
        model = PerShareCommission(rate_per_share=0.005, minimum=1.0)
        # 1000 shares * $0.005 = $5.00, above the $1 minimum
        assert model.charge(1000, 50.0) == pytest.approx(5.0)

    def test_per_share_applies_the_minimum_on_small_orders(self):
        model = PerShareCommission(rate_per_share=0.005, minimum=1.0)
        # 10 shares * $0.005 = $0.05, below the minimum
        assert model.charge(10, 50.0) == pytest.approx(1.0)

    def test_per_share_is_capped_as_a_share_of_notional(self):
        """A $1 minimum on a $20 order would be a 5% fee; brokers cap it."""
        model = PerShareCommission(rate_per_share=0.005, minimum=1.0, maximum_pct_of_notional=0.01)
        assert model.charge(10, 2.0) == pytest.approx(0.20)  # 1% of $20

    def test_percent_commission_is_basis_points_of_notional(self):
        model = PercentCommission(bps=5.0)
        assert model.charge(100, 200.0) == pytest.approx(100 * 200.0 * 5 * BPS)

    def test_commission_is_never_negative(self):
        for model in (PerShareCommission(), PercentCommission(), NoCommission()):
            assert model.charge(100, 10.0) >= 0.0

    def test_commission_ignores_the_sign_of_quantity(self):
        model = PerShareCommission()
        assert model.charge(-500, 20.0) == model.charge(500, 20.0)


class TestSlippage:
    def test_no_slippage_returns_the_reference(self):
        assert NoSlippage().fill_price(OrderSide.BUY, 100.0, 10) == 100.0

    def test_a_buy_never_fills_below_the_reference(self):
        fill = FixedBpsSlippage(10.0).fill_price(OrderSide.BUY, 100.0, 10)
        assert fill > 100.0
        assert fill == pytest.approx(100.0 * (1 + 10 * BPS))

    def test_a_sell_never_fills_above_the_reference(self):
        fill = FixedBpsSlippage(10.0).fill_price(OrderSide.SELL, 100.0, 10)
        assert fill < 100.0
        assert fill == pytest.approx(100.0 * (1 - 10 * BPS))

    def test_slippage_is_symmetric_around_the_reference(self):
        model = FixedBpsSlippage(25.0)
        buy = model.fill_price(OrderSide.BUY, 50.0, 1)
        sell = model.fill_price(OrderSide.SELL, 50.0, 1)
        assert buy - 50.0 == pytest.approx(50.0 - sell)


class TestSquareRootImpact:
    def test_impact_grows_with_size(self):
        model = SquareRootImpactSlippage()
        small = model.fill_price(OrderSide.BUY, 100.0, 1_000, bar_volume=1_000_000, volatility=0.02)
        large = model.fill_price(
            OrderSide.BUY, 100.0, 100_000, bar_volume=1_000_000, volatility=0.02
        )
        assert large > small

    def test_impact_grows_sub_linearly(self):
        """Quadrupling size should roughly double impact, not quadruple it."""
        model = SquareRootImpactSlippage(half_spread_bps=0.0, impact_coefficient=1.0)
        one = (
            model.fill_price(OrderSide.BUY, 100.0, 10_000, bar_volume=1_000_000, volatility=0.02)
            - 100.0
        )
        four = (
            model.fill_price(OrderSide.BUY, 100.0, 40_000, bar_volume=1_000_000, volatility=0.02)
            - 100.0
        )
        assert four == pytest.approx(2.0 * one, rel=1e-9)

    def test_participation_is_capped(self):
        """Beyond the cap, extra size does not keep adding impact in this model."""
        model = SquareRootImpactSlippage(max_participation=0.1)
        at_cap = model.fill_price(
            OrderSide.BUY, 100.0, 100_000, bar_volume=1_000_000, volatility=0.02
        )
        beyond = model.fill_price(
            OrderSide.BUY, 100.0, 900_000, bar_volume=1_000_000, volatility=0.02
        )
        assert beyond == pytest.approx(at_cap)

    def test_missing_volume_degrades_to_the_half_spread(self):
        model = SquareRootImpactSlippage(half_spread_bps=3.0)
        fill = model.fill_price(OrderSide.BUY, 100.0, 10_000, bar_volume=None)
        assert fill == pytest.approx(100.0 * (1 + 3.0 * BPS))

    def test_zero_volume_does_not_divide_by_zero(self):
        model = SquareRootImpactSlippage(half_spread_bps=3.0)
        fill = model.fill_price(OrderSide.BUY, 100.0, 10_000, bar_volume=0.0)
        assert fill == pytest.approx(100.0 * (1 + 3.0 * BPS))


class TestCostModel:
    def test_describe_round_trips_to_plain_types(self):
        described = CostModel(PerShareCommission(), FixedBpsSlippage()).describe()
        assert described["commission"]["model"] == "PerShareCommission"
        assert described["slippage"]["model"] == "FixedBpsSlippage"

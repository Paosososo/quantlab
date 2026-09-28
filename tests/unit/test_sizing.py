"""Position sizing."""

from __future__ import annotations

import pytest

from quantlab.backtesting.sizing import TargetWeightSizer, VolatilityTargetSizer
from quantlab.exceptions import BacktestError

pytestmark = pytest.mark.unit


class TestTargetWeightSizer:
    def test_weight_translates_to_shares(self):
        sizer = TargetWeightSizer(allow_fractional=True, max_participation=None)
        result = sizer.target_quantity(symbol="A", target_weight=0.5, price=50.0, equity=10_000.0)
        assert result.quantity == pytest.approx(100.0)

    def test_whole_share_rounding_moves_toward_zero(self):
        """Rounding must never increase exposure beyond the target."""
        sizer = TargetWeightSizer(allow_fractional=False, max_participation=None)
        long = sizer.target_quantity(symbol="A", target_weight=1.0, price=30.0, equity=1_000.0)
        short = sizer.target_quantity(symbol="A", target_weight=-1.0, price=30.0, equity=1_000.0)
        assert long.quantity == 33.0
        assert short.quantity == -33.0

    def test_lot_size_is_respected(self):
        sizer = TargetWeightSizer(allow_fractional=False, lot_size=100, max_participation=None)
        result = sizer.target_quantity(symbol="A", target_weight=1.0, price=10.0, equity=12_345.0)
        assert result.quantity == 1_200.0

    def test_max_weight_caps_the_target(self):
        sizer = TargetWeightSizer(allow_fractional=True, max_weight=0.5, max_participation=None)
        result = sizer.target_quantity(symbol="A", target_weight=2.0, price=10.0, equity=1_000.0)
        assert result.quantity == pytest.approx(50.0)
        assert result.capped_by == "max_weight"

    def test_participation_cap_limits_the_order(self):
        sizer = TargetWeightSizer(allow_fractional=True, max_participation=0.1)
        result = sizer.target_quantity(
            symbol="A", target_weight=1.0, price=1.0, equity=1_000_000.0, bar_volume=10_000.0
        )
        assert result.quantity == pytest.approx(1_000.0)
        assert result.capped_by == "max_participation"

    def test_non_positive_price_raises(self):
        with pytest.raises(BacktestError):
            TargetWeightSizer().target_quantity(
                symbol="A", target_weight=1.0, price=0.0, equity=1_000.0
            )


class TestVolatilityTargetSizer:
    def test_low_volatility_scales_the_position_up(self):
        sizer = VolatilityTargetSizer(
            annual_vol_target=0.10, allow_fractional=True, max_participation=None
        )
        result = sizer.target_quantity(
            symbol="A", target_weight=1.0, price=100.0, equity=100_000.0, volatility=0.05
        )
        # 10% target / 5% realised => 2x weight => 2000 shares
        assert result.quantity == pytest.approx(2_000.0)

    def test_high_volatility_scales_the_position_down(self):
        sizer = VolatilityTargetSizer(
            annual_vol_target=0.10, allow_fractional=True, max_participation=None
        )
        result = sizer.target_quantity(
            symbol="A", target_weight=1.0, price=100.0, equity=100_000.0, volatility=0.40
        )
        assert result.quantity == pytest.approx(250.0)

    def test_leverage_cap_binds_in_a_quiet_market(self):
        sizer = VolatilityTargetSizer(
            annual_vol_target=0.10,
            max_leverage=2.0,
            allow_fractional=True,
            max_participation=None,
        )
        result = sizer.target_quantity(
            symbol="A", target_weight=1.0, price=100.0, equity=100_000.0, volatility=0.001
        )
        assert result.quantity == pytest.approx(2_000.0)
        assert result.capped_by == "max_leverage"

    def test_missing_volatility_falls_back(self):
        sizer = VolatilityTargetSizer(
            annual_vol_target=0.10,
            fallback_volatility=0.20,
            allow_fractional=True,
            max_participation=None,
        )
        result = sizer.target_quantity(
            symbol="A", target_weight=1.0, price=100.0, equity=100_000.0, volatility=None
        )
        assert result.quantity == pytest.approx(500.0)

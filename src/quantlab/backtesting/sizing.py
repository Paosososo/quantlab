"""Position sizing: turning target weights into share counts.

Kept separate from strategy logic because sizing is a risk decision, not an
alpha decision.  The same signal can be sized flat, scaled to a volatility
target or capped by liquidity, and being able to swap that independently is what
lets you ask "is this strategy's edge in the signal or in the sizing?".
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from quantlab.exceptions import BacktestError


@dataclass(frozen=True, slots=True)
class SizingResult:
    quantity: float
    capped_by: str | None = None


class PositionSizer(ABC):
    @abstractmethod
    def target_quantity(
        self,
        *,
        symbol: str,
        target_weight: float,
        price: float,
        equity: float,
        volatility: float | None = None,
        bar_volume: float | None = None,
    ) -> SizingResult: ...

    def describe(self) -> dict[str, object]:
        return {"model": type(self).__name__}

    @staticmethod
    def _round(quantity: float, *, allow_fractional: bool, lot_size: int) -> float:
        if allow_fractional:
            return quantity
        if lot_size <= 0:
            raise BacktestError("lot size must be positive", lot_size=lot_size)
        # Round toward zero so sizing never accidentally increases exposure
        # beyond the target because of rounding.
        lots = math.trunc(quantity / lot_size)
        return float(lots * lot_size)


@dataclass(frozen=True, slots=True)
class TargetWeightSizer(PositionSizer):
    """Straight translation of a weight into shares: ``w * equity / price``.

    ``max_participation`` caps an order at a fraction of the bar's volume.  A
    backtest that buys 40% of a day's volume is not describing a trade anyone
    could have made, and this cap is the difference between a strategy that
    scales and one that only works on paper.
    """

    allow_fractional: bool = False
    lot_size: int = 1
    max_weight: float = 1.0
    max_participation: float | None = 0.10

    def target_quantity(
        self,
        *,
        symbol: str,
        target_weight: float,
        price: float,
        equity: float,
        volatility: float | None = None,  # noqa: ARG002
        bar_volume: float | None = None,
    ) -> SizingResult:
        if price <= 0:
            raise BacktestError("cannot size a position at a non-positive price", symbol=symbol)
        capped_by: str | None = None
        weight = target_weight
        if abs(weight) > self.max_weight:
            weight = math.copysign(self.max_weight, weight)
            capped_by = "max_weight"

        quantity = weight * equity / price

        if self.max_participation is not None and bar_volume and bar_volume > 0:
            limit = self.max_participation * bar_volume
            if abs(quantity) > limit:
                quantity = math.copysign(limit, quantity)
                capped_by = "max_participation"

        return SizingResult(
            self._round(quantity, allow_fractional=self.allow_fractional, lot_size=self.lot_size),
            capped_by,
        )

    def describe(self) -> dict[str, object]:
        return {
            "model": "TargetWeightSizer",
            "allow_fractional": self.allow_fractional,
            "lot_size": self.lot_size,
            "max_weight": self.max_weight,
            "max_participation": self.max_participation,
        }


@dataclass(frozen=True, slots=True)
class VolatilityTargetSizer(PositionSizer):
    """Scale each position so its standalone risk hits ``annual_vol_target``.

    ``scaled_weight = target_weight * vol_target / realised_vol``.  Realised
    volatility must be estimated from data available at the decision time; the
    engine passes a point-in-time estimate, never a full-sample one.

    ``max_leverage`` stops the scaler from taking absurd positions in a quiet
    market, which is the standard failure mode of naive vol targeting.
    """

    annual_vol_target: float = 0.10
    max_leverage: float = 2.0
    allow_fractional: bool = False
    lot_size: int = 1
    max_participation: float | None = 0.10
    fallback_volatility: float = 0.20

    def target_quantity(
        self,
        *,
        symbol: str,
        target_weight: float,
        price: float,
        equity: float,
        volatility: float | None = None,
        bar_volume: float | None = None,
    ) -> SizingResult:
        if price <= 0:
            raise BacktestError("cannot size a position at a non-positive price", symbol=symbol)
        sigma = volatility if volatility and volatility > 1e-8 else self.fallback_volatility
        scale = self.annual_vol_target / sigma
        weight = target_weight * scale
        capped_by: str | None = None
        if abs(weight) > self.max_leverage:
            weight = math.copysign(self.max_leverage, weight)
            capped_by = "max_leverage"

        quantity = weight * equity / price
        if self.max_participation is not None and bar_volume and bar_volume > 0:
            limit = self.max_participation * bar_volume
            if abs(quantity) > limit:
                quantity = math.copysign(limit, quantity)
                capped_by = "max_participation"
        return SizingResult(
            self._round(quantity, allow_fractional=self.allow_fractional, lot_size=self.lot_size),
            capped_by,
        )

    def describe(self) -> dict[str, object]:
        return {
            "model": "VolatilityTargetSizer",
            "annual_vol_target": self.annual_vol_target,
            "max_leverage": self.max_leverage,
            "allow_fractional": self.allow_fractional,
            "max_participation": self.max_participation,
        }


__all__ = ["PositionSizer", "SizingResult", "TargetWeightSizer", "VolatilityTargetSizer"]

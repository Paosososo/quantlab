"""Transaction cost models.

Costs are the difference between a paper strategy and a real one, and they are
where most public backtests quietly cheat.  Three components are modelled
separately because they behave differently and should be tunable independently:

**Commission** -- an explicit fee.  Scales with shares or notional and usually
has a floor.

**Slippage** -- the gap between the price you referenced and the price you got,
for reasons other than your own size.  Bid-ask spread lives here.

**Market impact** -- the price move your own order causes.  Modelled as the
square root of participation rate, which is the standard empirical form
(impact grows with size but sub-linearly).

The sign convention is enforced in one place: a buy always fills at or above
the reference price, a sell at or below.  Getting this backwards turns costs
into a source of alpha, and it is a surprisingly common bug, so there is a test
for it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from quantlab.db.enums import OrderSide
from quantlab.exceptions import BacktestError

BPS = 1e-4


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------
class CommissionModel(ABC):
    @abstractmethod
    def charge(self, quantity: float, price: float) -> float:
        """Fee for trading ``quantity`` shares at ``price``.  Always >= 0."""

    def describe(self) -> dict[str, float | str]:
        return {"model": type(self).__name__}


@dataclass(frozen=True, slots=True)
class NoCommission(CommissionModel):
    def charge(self, quantity: float, price: float) -> float:  # noqa: ARG002
        return 0.0


@dataclass(frozen=True, slots=True)
class PerShareCommission(CommissionModel):
    """US retail-broker shape: cents per share with a per-order minimum."""

    rate_per_share: float = 0.005
    minimum: float = 1.0
    maximum_pct_of_notional: float = 0.01

    def charge(self, quantity: float, price: float) -> float:
        raw = abs(quantity) * self.rate_per_share
        fee = max(raw, self.minimum)
        cap = abs(quantity) * price * self.maximum_pct_of_notional
        return float(min(fee, cap)) if cap > 0 else float(fee)

    def describe(self) -> dict[str, float | str]:
        return {
            "model": "PerShareCommission",
            "rate_per_share": self.rate_per_share,
            "minimum": self.minimum,
            "maximum_pct_of_notional": self.maximum_pct_of_notional,
        }


@dataclass(frozen=True, slots=True)
class PercentCommission(CommissionModel):
    """Basis points of notional, with an optional floor.  Common outside the US."""

    bps: float = 5.0
    minimum: float = 0.0

    def charge(self, quantity: float, price: float) -> float:
        return float(max(abs(quantity) * price * self.bps * BPS, self.minimum))

    def describe(self) -> dict[str, float | str]:
        return {"model": "PercentCommission", "bps": self.bps, "minimum": self.minimum}


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------
class SlippageModel(ABC):
    @abstractmethod
    def fill_price(
        self,
        side: OrderSide,
        reference_price: float,
        quantity: float,
        *,
        bar_volume: float | None = None,
        volatility: float | None = None,
    ) -> float:
        """Price actually paid or received.  Never better than ``reference_price``."""

    def describe(self) -> dict[str, float | str]:
        return {"model": type(self).__name__}

    @staticmethod
    def _apply(side: OrderSide, reference_price: float, penalty_fraction: float) -> float:
        """Apply a non-negative penalty in the direction that hurts."""
        if penalty_fraction < 0:
            raise BacktestError("slippage penalty must be non-negative", value=penalty_fraction)
        direction = 1.0 if side is OrderSide.BUY else -1.0
        return reference_price * (1.0 + direction * penalty_fraction)


@dataclass(frozen=True, slots=True)
class NoSlippage(SlippageModel):
    def fill_price(
        self,
        side: OrderSide,  # noqa: ARG002
        reference_price: float,
        quantity: float,  # noqa: ARG002
        *,
        bar_volume: float | None = None,  # noqa: ARG002
        volatility: float | None = None,  # noqa: ARG002
    ) -> float:
        return reference_price


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage(SlippageModel):
    """Constant half-spread in basis points.  The simplest defensible model."""

    bps: float = 5.0

    def fill_price(
        self,
        side: OrderSide,
        reference_price: float,
        quantity: float,  # noqa: ARG002
        *,
        bar_volume: float | None = None,  # noqa: ARG002
        volatility: float | None = None,  # noqa: ARG002
    ) -> float:
        return self._apply(side, reference_price, self.bps * BPS)


@dataclass(frozen=True, slots=True)
class SquareRootImpactSlippage(SlippageModel):
    """Half-spread plus square-root market impact.

    ``impact = coefficient * volatility * sqrt(participation)`` where
    ``participation = |quantity| / bar_volume``.  The square-root form is the
    standard empirical result (Almgren and others): doubling order size less
    than doubles impact.  The coefficient is a free parameter; the default of
    1.0 with daily volatility is a conventional starting point, not a
    calibrated estimate, and the docs say so.

    When ``bar_volume`` is missing or zero the model degrades to the fixed
    half-spread rather than dividing by zero or silently charging nothing.
    """

    half_spread_bps: float = 2.5
    impact_coefficient: float = 1.0
    default_volatility: float = 0.02
    max_participation: float = 0.25

    def fill_price(
        self,
        side: OrderSide,
        reference_price: float,
        quantity: float,
        *,
        bar_volume: float | None = None,
        volatility: float | None = None,
    ) -> float:
        penalty = self.half_spread_bps * BPS
        if bar_volume and bar_volume > 0:
            participation = min(abs(quantity) / bar_volume, self.max_participation)
            sigma = volatility if volatility and volatility > 0 else self.default_volatility
            penalty += self.impact_coefficient * sigma * math.sqrt(participation)
        return self._apply(side, reference_price, penalty)

    def describe(self) -> dict[str, float | str]:
        return {
            "model": "SquareRootImpactSlippage",
            "half_spread_bps": self.half_spread_bps,
            "impact_coefficient": self.impact_coefficient,
            "default_volatility": self.default_volatility,
            "max_participation": self.max_participation,
        }


@dataclass(frozen=True, slots=True)
class CostModel:
    """The pair of models the engine actually consults."""

    commission: CommissionModel = PerShareCommission()
    slippage: SlippageModel = FixedBpsSlippage()

    def describe(self) -> dict[str, object]:
        return {"commission": self.commission.describe(), "slippage": self.slippage.describe()}


ZERO_COST = CostModel(commission=NoCommission(), slippage=NoSlippage())
"""No-cost model.  Used only to verify engine arithmetic against closed-form
results in tests, never for reported research."""


__all__ = [
    "BPS",
    "ZERO_COST",
    "CommissionModel",
    "CostModel",
    "FixedBpsSlippage",
    "NoCommission",
    "NoSlippage",
    "PerShareCommission",
    "PercentCommission",
    "SlippageModel",
    "SquareRootImpactSlippage",
]

"""Pydantic request and response models.

Separate from the ORM models on purpose.  Returning SQLAlchemy objects directly
couples the public contract to the storage layer, so a harmless column rename
becomes a breaking API change, and it makes it easy to leak internal fields by
accident.  A response model is a deliberate statement of what the API promises.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer

T = TypeVar("T")


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------
class Page(ApiModel, Generic[T]):
    """A page of results.

    ``total`` is deliberately optional: counting a large table on every request
    is expensive, and most callers only need to know whether there is more.
    """

    items: list[T]
    limit: int
    offset: int
    returned: int
    total: int | None = None


class ErrorDetail(ApiModel):
    error: str = Field(description="Machine-readable error class")
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(ApiModel):
    status: str
    version: str
    environment: str
    database: str
    checked_at: dt.datetime


class ReadinessResponse(ApiModel):
    ready: bool
    database: bool
    migrations_applied: bool | None = None
    detail: str | None = None


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
class AssetOut(ApiModel):
    id: int
    symbol: str
    name: str | None = None
    asset_class: str
    exchange: str | None = None
    currency: str
    status: str
    listed_on: dt.date | None = None
    delisted_on: dt.date | None = None


class AssetDetail(AssetOut):
    first_price_ts: dt.datetime | None = None
    last_price_ts: dt.datetime | None = None
    n_prices: int = 0


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
class PriceOut(ApiModel):
    symbol: str
    ts: dt.datetime
    available_at: dt.datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None = None
    volume: float | None = None

    @field_serializer("open", "high", "low", "close", "adj_close", "volume")
    def _serialise_decimal(self, value: float | Decimal | None) -> float | None:
        return None if value is None else float(value)


class ReturnOut(ApiModel):
    symbol: str
    ts: dt.datetime
    available_at: dt.datetime
    horizon_days: int
    direction: str
    method: str
    value: float


class EconomicObservationOut(ApiModel):
    code: str
    ts: dt.datetime
    available_at: dt.datetime
    value: float | None = None


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
class FeatureSetOut(ApiModel):
    id: int
    name: str
    version: int
    spec_hash: str
    description: str | None = None
    n_features: int


class FeatureValueOut(ApiModel):
    symbol: str
    name: str
    ts: dt.datetime
    available_at: dt.datetime
    value: float | None = None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ModelRunOut(ApiModel):
    id: int
    name: str
    model_type: str
    target_name: str
    config_hash: str
    data_fingerprint: str | None = None
    code_version: str | None = None
    seed: int
    validation_scheme: str
    status: str
    started_at: dt.datetime
    finished_at: dt.datetime | None = None
    metrics: dict[str, Any] | None = None


class PredictionOut(ApiModel):
    symbol: str
    ts: dt.datetime
    target_ts: dt.datetime
    fold: int
    split: str
    y_pred: float
    y_true: float | None = None
    y_proba: float | None = None


class ModelComparisonRow(ApiModel):
    model_run_id: int
    model_type: str
    rmse: float | None = None
    r2_oos_vs_zero: float | None = None
    information_coefficient: float | None = None
    directional_accuracy: float | None = None
    n: int | None = None


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------
class BacktestOut(ApiModel):
    id: int
    strategy_name: str
    strategy_kind: str
    start_ts: dt.datetime
    end_ts: dt.datetime
    initial_cash: float
    execution_timing: str
    config_hash: str
    status: str
    benchmark_symbol: str | None = None
    metrics: dict[str, Any] | None = None

    @field_serializer("initial_cash")
    def _serialise_cash(self, value: float | Decimal) -> float:
        return float(value)


class EquityPoint(ApiModel):
    ts: dt.datetime
    equity: float
    cash: float
    positions_value: float
    period_return: float | None = None
    drawdown: float | None = None
    benchmark_equity: float | None = None
    leverage: float
    n_positions: int
    turnover: float | None = None


class TradeOut(ApiModel):
    symbol: str
    decision_ts: dt.datetime
    execution_ts: dt.datetime
    side: str
    quantity: float
    reference_price: float
    fill_price: float
    commission: float
    slippage_cost: float
    notional: float
    realised_pnl: float | None = None


class PerformanceSummary(ApiModel):
    backtest_id: int
    strategy_name: str
    metrics: dict[str, Any]
    risk: dict[str, Any]
    sharpe_inference: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Analytics requests
# ---------------------------------------------------------------------------
class OptimisationRequest(ApiModel):
    symbols: list[str] = Field(min_length=2, max_length=50)
    method: str = Field(
        default="minimum_variance",
        description="equal_weight | inverse_volatility | minimum_variance | risk_parity",
    )
    lookback_days: int = Field(default=756, ge=60, le=5_000)
    shrinkage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Manual shrinkage intensity; omit for the Ledoit-Wolf estimate",
    )
    as_of: dt.datetime | None = Field(
        default=None,
        description=(
            "Point-in-time cutoff.  Only observations published at or before this "
            "instant are used, so the result is what the optimiser would have "
            "produced on that date."
        ),
    )


class OptimisationResponse(ApiModel):
    method: str
    as_of: dt.datetime | None
    n_observations: int
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    risk_contributions: dict[str, float]
    diversification_ratio: float | None = None
    converged: bool


class RiskRequest(ApiModel):
    symbol: str
    lookback_days: int = Field(default=756, ge=60, le=5_000)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    as_of: dt.datetime | None = None


__all__ = [
    "ApiModel",
    "AssetDetail",
    "AssetOut",
    "BacktestOut",
    "EconomicObservationOut",
    "EquityPoint",
    "ErrorDetail",
    "FeatureSetOut",
    "FeatureValueOut",
    "HealthResponse",
    "ModelComparisonRow",
    "ModelRunOut",
    "OptimisationRequest",
    "OptimisationResponse",
    "Page",
    "PerformanceSummary",
    "PredictionOut",
    "PriceOut",
    "ReadinessResponse",
    "ReturnOut",
    "RiskRequest",
    "TradeOut",
]

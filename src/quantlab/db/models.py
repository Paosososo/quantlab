"""ORM models.

Schema notes worth defending
----------------------------
*Every fact table carries both ``ts`` and ``available_at``.*  See
:mod:`quantlab.timeutils` for why.  ``CHECK (available_at >= ts)`` is enforced
by the database, so a loader bug cannot quietly insert a row that claims to have
been knowable before the period it describes.

*Universe membership is temporal.*  ``universe_members`` has ``valid_from`` /
``valid_to`` so a backtest can ask "which symbols were in this index on
2011-06-30?" rather than "which symbols are in it today?".  Asking the second
question is survivorship bias, and it is the reason naive backtests of index
strategies look better than reality.

*Features are stored long, not wide.*  ``(feature_set_id, asset_id, ts, name)``
instead of one column per feature.  Adding a feature then costs an insert, not
a migration, which matters when the research loop is the point of the system.
The cost is a larger table and a pivot on read; the pivot happens once per query
in :mod:`quantlab.db.repository` and is cheap relative to model fitting.

*Enums are VARCHAR + CHECK, not native PostgreSQL types.*  See
:mod:`quantlab.db.base`.

*Prices and cash are NUMERIC; statistics are DOUBLE PRECISION.*  Accounting
quantities should be exact in the system of record.  Returns, feature values
and metrics are estimates where float range beats the last decimal place.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from quantlab.db.base import Base, Money, Quantity, TimestampMixin, TZDateTime
from quantlab.db.enums import (
    AssetClass,
    AssetStatus,
    CheckStatus,
    ExecutionTiming,
    Frequency,
    OrderSide,
    OrderStatus,
    OrderType,
    ReturnDirection,
    ReturnMethod,
    RunStatus,
    SplitKind,
    enum_values,
)

#: JSONB on PostgreSQL, plain JSON elsewhere, so unit tests can run on SQLite.
JSONType = JSON().with_variant(JSONB(), "postgresql")


def _enum(enum_cls: type, name: str) -> Enum:
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        length=32,
        values_callable=enum_values,
        validate_strings=True,
    )


# ===========================================================================
# Reference data
# ===========================================================================
class Asset(Base, TimestampMixin):
    """A tradable or observable instrument.

    ``listed_on`` / ``delisted_on`` exist so that survivorship can be reasoned
    about.  We cannot fully solve survivorship with free data (delisted tickers
    generally are not downloadable), but recording what we know beats pretending
    the problem does not exist.
    """

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str | None] = mapped_column(String(200))
    asset_class: Mapped[AssetClass] = mapped_column(
        _enum(AssetClass, "asset_class"), nullable=False
    )
    exchange: Mapped[str | None] = mapped_column(String(32))
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    session_name: Mapped[str] = mapped_column(String(32), nullable=False, default="us_equity")
    status: Mapped[AssetStatus] = mapped_column(
        _enum(AssetStatus, "asset_status"), nullable=False, default=AssetStatus.ACTIVE
    )
    listed_on: Mapped[dt.date | None] = mapped_column()
    delisted_on: Mapped[dt.date | None] = mapped_column()
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONType)

    prices: Mapped[list[Price]] = relationship(back_populates="asset", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("symbol", name="uq_assets_symbol"),
        CheckConstraint("currency = upper(currency)", name="currency_upper"),
        CheckConstraint(
            "delisted_on IS NULL OR listed_on IS NULL OR delisted_on >= listed_on",
            name="delist_after_list",
        ),
        Index("ix_assets_asset_class_status", "asset_class", "status"),
    )


class Universe(Base, TimestampMixin):
    """A named, point-in-time set of assets (an index, a screen, a test bed)."""

    __tablename__ = "universes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)

    members: Mapped[list[UniverseMember]] = relationship(
        back_populates="universe", cascade="all, delete-orphan"
    )


class UniverseMember(Base):
    """Membership of an asset in a universe over a half-open interval.

    ``valid_to IS NULL`` means "still a member".  Queries take the form
    ``valid_from <= :as_of AND (valid_to IS NULL OR valid_to > :as_of)``.
    """

    __tablename__ = "universe_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    universe_id: Mapped[int] = mapped_column(
        ForeignKey("universes.id", ondelete="CASCADE"), nullable=False
    )
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    valid_from: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    valid_to: Mapped[dt.datetime | None] = mapped_column(TZDateTime)

    universe: Mapped[Universe] = relationship(back_populates="members")
    asset: Mapped[Asset] = relationship()

    __table_args__ = (
        UniqueConstraint("universe_id", "asset_id", "valid_from", name="uq_universe_members_slice"),
        CheckConstraint("valid_to IS NULL OR valid_to > valid_from", name="interval_ordered"),
        Index("ix_universe_members_universe_id_valid_from", "universe_id", "valid_from"),
    )


# ===========================================================================
# Market data
# ===========================================================================
class Price(Base):
    """One daily OHLCV bar.

    Unique on ``(asset_id, ts)``: the table holds one canonical bar per asset
    per session.  Which provider supplied it is recorded in ``source`` and the
    untouched payload from every provider is kept in the raw layer on disk, so
    the canonical row can always be rebuilt or reconciled.
    """

    __tablename__ = "prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    available_at: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)

    open: Mapped[Decimal] = mapped_column(Money, nullable=False)
    high: Mapped[Decimal] = mapped_column(Money, nullable=False)
    low: Mapped[Decimal] = mapped_column(Money, nullable=False)
    close: Mapped[Decimal] = mapped_column(Money, nullable=False)
    adj_close: Mapped[Decimal | None] = mapped_column(Money)
    volume: Mapped[Decimal | None] = mapped_column(Quantity)

    source: Mapped[str] = mapped_column(String(32), nullable=False)
    ingested_at: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)

    asset: Mapped[Asset] = relationship(back_populates="prices")

    __table_args__ = (
        UniqueConstraint("asset_id", "ts", name="uq_prices_asset_id_ts"),
        CheckConstraint("high >= low", name="high_ge_low"),
        CheckConstraint("high >= open AND high >= close", name="high_is_max"),
        CheckConstraint("low <= open AND low <= close", name="low_is_min"),
        CheckConstraint("open > 0 AND high > 0 AND low > 0 AND close > 0", name="positive_prices"),
        CheckConstraint("volume IS NULL OR volume >= 0", name="volume_non_negative"),
        CheckConstraint("available_at >= ts", name="available_after_ts"),
        Index("ix_prices_ts", "ts"),
        Index("ix_prices_asset_id_available_at", "asset_id", "available_at"),
    )


class Return(Base):
    """A derived return observation.

    ``direction`` separates trailing returns (safe as features) from forward
    returns (labels).  A forward return's ``available_at`` is the availability
    of the *last* price in its window, which is what stops a label leaking into
    the feature set through a careless join.
    """

    __tablename__ = "returns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    available_at: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    direction: Mapped[ReturnDirection] = mapped_column(
        _enum(ReturnDirection, "return_direction"), nullable=False
    )
    method: Mapped[ReturnMethod] = mapped_column(
        _enum(ReturnMethod, "return_method"), nullable=False
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)
    price_field: Mapped[str] = mapped_column(String(16), nullable=False, default="adj_close")

    asset: Mapped[Asset] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "asset_id",
            "ts",
            "horizon_days",
            "direction",
            "method",
            "price_field",
            name="uq_returns_key",
        ),
        CheckConstraint("horizon_days > 0", name="positive_horizon"),
        CheckConstraint("available_at >= ts", name="available_after_ts"),
        Index("ix_returns_asset_id_ts", "asset_id", "ts"),
    )


class EconomicSeries(Base, TimestampMixin):
    """Metadata for a macroeconomic series."""

    __tablename__ = "economic_series"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    title: Mapped[str | None] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="fred")
    frequency: Mapped[Frequency] = mapped_column(_enum(Frequency, "frequency"), nullable=False)
    units: Mapped[str | None] = mapped_column(String(64))
    #: Conservative worst-case delay between the period end and publication.
    #: Used to derive ``available_at`` because the free endpoint returns the
    #: latest vintage rather than the as-first-released value.
    publication_lag_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    seasonal_adjustment: Mapped[str | None] = mapped_column(String(32))

    observations: Mapped[list[EconomicObservation]] = relationship(
        back_populates="series", cascade="all, delete-orphan"
    )

    __table_args__ = (CheckConstraint("publication_lag_days >= 0", name="non_negative_lag"),)


class EconomicObservation(Base):
    __tablename__ = "economic_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("economic_series.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    available_at: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    ingested_at: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)

    series: Mapped[EconomicSeries] = relationship(back_populates="observations")

    __table_args__ = (
        UniqueConstraint("series_id", "ts", name="uq_economic_data_series_id_ts"),
        CheckConstraint("available_at >= ts", name="available_after_ts"),
        Index("ix_economic_data_series_id_available_at", "series_id", "available_at"),
    )


# ===========================================================================
# Features
# ===========================================================================
class FeatureSet(Base, TimestampMixin):
    """A named, versioned collection of feature definitions.

    ``spec_hash`` is a content hash of the serialised definitions.  Two runs
    with the same hash used the same feature logic, which is what makes a
    result reproducible without trusting a version string a human typed.
    """

    __tablename__ = "feature_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    spec_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    definition: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    values: Mapped[list[FeatureValue]] = relationship(
        back_populates="feature_set", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("name", "version", name="uq_feature_sets_name_version"),)


class FeatureValue(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature_set_id: Mapped[int] = mapped_column(
        ForeignKey("feature_sets.id", ondelete="CASCADE"), nullable=False
    )
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    available_at: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    value: Mapped[float | None] = mapped_column(Float)

    feature_set: Mapped[FeatureSet] = relationship(back_populates="values")
    asset: Mapped[Asset] = relationship()

    __table_args__ = (
        UniqueConstraint("feature_set_id", "asset_id", "name", "ts", name="uq_features_key"),
        CheckConstraint("available_at >= ts", name="available_after_ts"),
        Index("ix_features_lookup", "feature_set_id", "asset_id", "ts"),
        Index("ix_features_name_ts", "name", "ts"),
    )


# ===========================================================================
# Modelling
# ===========================================================================
class ModelRun(Base, TimestampMixin):
    """One fit-and-evaluate execution of one model configuration.

    The triple ``(config_hash, data_fingerprint, seed)`` is the reproducibility
    contract: same three values means the same numbers should come out.
    """

    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_set_id: Mapped[int | None] = mapped_column(
        ForeignKey("feature_sets.id", ondelete="SET NULL")
    )
    target_name: Mapped[str] = mapped_column(String(100), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    data_fingerprint: Mapped[str | None] = mapped_column(String(64))
    code_version: Mapped[str | None] = mapped_column(String(64))
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_scheme: Mapped[str] = mapped_column(String(64), nullable=False)
    train_start: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    train_end: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    test_start: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    test_end: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status"), nullable=False, default=RunStatus.RUNNING
    )
    started_at: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    finished_at: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(Text)

    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="model_run", cascade="all, delete-orphan"
    )

    # The four window columns describe the *span* of a run: the earliest and
    # latest training timestamps, and the earliest and latest test timestamps.
    #
    # There is deliberately no `CHECK (test_start >= train_end)` here.  An
    # earlier version had one, and an integration test caught it: in
    # walk-forward validation the aggregate spans legitimately overlap, because
    # fold 3 trains on data that fold 1 already tested on.  The constraint
    # encoded an assumption that holds only for a single holdout split.
    #
    # The ordering guarantee that actually matters is per fold, and it is
    # enforced where it is true: `assert_split_ordering` raises if any fold's
    # training data reaches into its test window, and `predictions` carries
    # `CHECK (target_ts > ts)` so no individual forecast can predict the past.
    __table_args__ = (
        Index("ix_model_runs_config_hash", "config_hash"),
        Index("ix_model_runs_model_type_status", "model_type", "status"),
    )


class Prediction(Base):
    """A single out-of-sample (or in-sample) forecast.

    ``ts`` is the *decision* time: the close of the bar whose information
    produced the forecast.  ``target_ts`` is when the predicted quantity is
    realised.  Keeping both makes it impossible to mistake one for the other
    when joining predictions back onto prices.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_run_id: Mapped[int] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), nullable=False
    )
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    target_ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    fold: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    split: Mapped[SplitKind] = mapped_column(_enum(SplitKind, "split_kind"), nullable=False)
    y_pred: Mapped[float] = mapped_column(Float, nullable=False)
    y_true: Mapped[float | None] = mapped_column(Float)
    y_proba: Mapped[float | None] = mapped_column(Float)

    model_run: Mapped[ModelRun] = relationship(back_populates="predictions")
    asset: Mapped[Asset] = relationship()

    __table_args__ = (
        UniqueConstraint("model_run_id", "asset_id", "ts", "fold", name="uq_predictions_key"),
        CheckConstraint("target_ts > ts", name="target_after_decision"),
        Index("ix_predictions_run_ts", "model_run_id", "ts"),
    )


# ===========================================================================
# Strategies and backtests
# ===========================================================================
class Strategy(Base, TimestampMixin):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    backtests: Mapped[list[Backtest]] = relationship(back_populates="strategy")

    __table_args__ = (UniqueConstraint("name", "params_hash", name="uq_strategies_name_params"),)


class Backtest(Base, TimestampMixin):
    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_id: Mapped[int] = mapped_column(
        ForeignKey("strategies.id", ondelete="CASCADE"), nullable=False
    )
    model_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_runs.id", ondelete="SET NULL")
    )
    universe_id: Mapped[int | None] = mapped_column(ForeignKey("universes.id", ondelete="SET NULL"))
    benchmark_asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("assets.id", ondelete="SET NULL")
    )

    start_ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    end_ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    initial_cash: Mapped[Decimal] = mapped_column(Money, nullable=False)
    execution_timing: Mapped[ExecutionTiming] = mapped_column(
        _enum(ExecutionTiming, "execution_timing"), nullable=False
    )
    cost_config: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    engine_config: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    code_version: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status"), nullable=False, default=RunStatus.RUNNING
    )
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    error: Mapped[str | None] = mapped_column(Text)

    strategy: Mapped[Strategy] = relationship(back_populates="backtests")
    trades: Mapped[list[Trade]] = relationship(
        back_populates="backtest", cascade="all, delete-orphan"
    )
    orders: Mapped[list[Order]] = relationship(
        back_populates="backtest", cascade="all, delete-orphan"
    )
    snapshots: Mapped[list[PortfolioSnapshot]] = relationship(
        back_populates="backtest", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("end_ts >= start_ts", name="end_after_start"),
        CheckConstraint("initial_cash > 0", name="positive_initial_cash"),
        Index("ix_backtests_config_hash", "config_hash"),
    )


class Order(Base):
    """An intent to trade, recorded at the moment it was decided.

    Persisting orders separately from fills is what makes the temporal claim
    auditable: ``decision_ts`` comes from the bar that produced the signal and
    ``execution_ts`` from the bar that filled it, and a database constraint
    requires the second to be strictly later than the first.
    """

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    backtest_id: Mapped[int] = mapped_column(
        ForeignKey("backtests.id", ondelete="CASCADE"), nullable=False
    )
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    client_order_id: Mapped[str] = mapped_column(String(64), nullable=False)
    decision_ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    execution_ts: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    side: Mapped[OrderSide] = mapped_column(_enum(OrderSide, "order_side"), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(_enum(OrderType, "order_type"), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Quantity, nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(Money)
    status: Mapped[OrderStatus] = mapped_column(_enum(OrderStatus, "order_status"), nullable=False)
    reject_reason: Mapped[str | None] = mapped_column(String(200))

    backtest: Mapped[Backtest] = relationship(back_populates="orders")

    __table_args__ = (
        UniqueConstraint("backtest_id", "client_order_id", name="uq_orders_backtest_client_id"),
        CheckConstraint("quantity > 0", name="positive_quantity"),
        CheckConstraint(
            "execution_ts IS NULL OR execution_ts > decision_ts", name="execution_after_decision"
        ),
        Index("ix_orders_backtest_decision_ts", "backtest_id", "decision_ts"),
    )


class Trade(Base):
    """A fill.  One order may produce one fill in this engine (no partials yet)."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    backtest_id: Mapped[int] = mapped_column(
        ForeignKey("backtests.id", ondelete="CASCADE"), nullable=False
    )
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id", ondelete="SET NULL"))
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    decision_ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    execution_ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    side: Mapped[OrderSide] = mapped_column(_enum(OrderSide, "order_side"), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Quantity, nullable=False)
    reference_price: Mapped[Decimal] = mapped_column(Money, nullable=False)
    fill_price: Mapped[Decimal] = mapped_column(Money, nullable=False)
    commission: Mapped[Decimal] = mapped_column(Money, nullable=False, default=0)
    slippage_cost: Mapped[Decimal] = mapped_column(Money, nullable=False, default=0)
    notional: Mapped[Decimal] = mapped_column(Money, nullable=False)
    realised_pnl: Mapped[Decimal | None] = mapped_column(Money)

    backtest: Mapped[Backtest] = relationship(back_populates="trades")

    __table_args__ = (
        CheckConstraint("quantity > 0", name="positive_quantity"),
        CheckConstraint("fill_price > 0 AND reference_price > 0", name="positive_prices"),
        CheckConstraint("commission >= 0 AND slippage_cost >= 0", name="non_negative_costs"),
        CheckConstraint("execution_ts > decision_ts", name="execution_after_decision"),
        Index("ix_trades_backtest_execution_ts", "backtest_id", "execution_ts"),
    )


class PortfolioSnapshot(Base):
    """End-of-bar portfolio state.  One row per bar per backtest."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    backtest_id: Mapped[int] = mapped_column(
        ForeignKey("backtests.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    cash: Mapped[Decimal] = mapped_column(Money, nullable=False)
    positions_value: Mapped[Decimal] = mapped_column(Money, nullable=False)
    equity: Mapped[Decimal] = mapped_column(Money, nullable=False)
    gross_exposure: Mapped[Decimal] = mapped_column(Money, nullable=False)
    net_exposure: Mapped[Decimal] = mapped_column(Money, nullable=False)
    leverage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    n_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    period_return: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float | None] = mapped_column(Float)
    benchmark_equity: Mapped[Decimal | None] = mapped_column(Money)
    turnover: Mapped[float | None] = mapped_column(Float)

    backtest: Mapped[Backtest] = relationship(back_populates="snapshots")

    __table_args__ = (
        UniqueConstraint("backtest_id", "ts", name="uq_portfolio_snapshots_backtest_ts"),
        Index("ix_portfolio_snapshots_backtest_ts", "backtest_id", "ts"),
    )


class PositionSnapshot(Base):
    """Per-asset holding at a snapshot time.  Optional; written when enabled."""

    __tablename__ = "position_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    backtest_id: Mapped[int] = mapped_column(
        ForeignKey("backtests.id", ondelete="CASCADE"), nullable=False
    )
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Quantity, nullable=False)
    price: Mapped[Decimal] = mapped_column(Money, nullable=False)
    market_value: Mapped[Decimal] = mapped_column(Money, nullable=False)
    weight: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("backtest_id", "asset_id", "ts", name="uq_position_snapshots_key"),
        Index("ix_position_snapshots_backtest_ts", "backtest_id", "ts"),
    )


# ===========================================================================
# Operational metadata
# ===========================================================================
class IngestionRun(Base):
    """Bookkeeping for one execution of one ingestion job.

    ``watermark_to`` is what makes incremental loading possible and idempotent:
    the next run asks the database "what is the newest data I already have for
    this (provider, dataset, entity)?" rather than trusting a scheduler date.
    """

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    entity: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    finished_at: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    status: Mapped[RunStatus] = mapped_column(_enum(RunStatus, "run_status"), nullable=False)
    rows_fetched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_written: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    watermark_from: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    watermark_to: Mapped[dt.datetime | None] = mapped_column(TZDateTime)
    raw_path: Mapped[str | None] = mapped_column(String(500))
    payload_sha256: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_ingestion_runs_lookup", "provider", "dataset", "entity", "started_at"),
    )


class DataQualityCheck(Base):
    """Result of one validation rule against one dataset slice."""

    __tablename__ = "data_quality_checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    entity: Mapped[str | None] = mapped_column(String(64))
    check_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[CheckStatus] = mapped_column(_enum(CheckStatus, "check_status"), nullable=False)
    observed: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    message: Mapped[str | None] = mapped_column(Text)
    checked_at: Mapped[dt.datetime] = mapped_column(TZDateTime, nullable=False)
    ingestion_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL")
    )

    __table_args__ = (
        Index("ix_data_quality_checks_dataset_checked_at", "dataset", "checked_at"),
        Index("ix_data_quality_checks_status", "status"),
    )


__all__ = [
    "Asset",
    "Backtest",
    "DataQualityCheck",
    "EconomicObservation",
    "EconomicSeries",
    "FeatureSet",
    "FeatureValue",
    "IngestionRun",
    "ModelRun",
    "Order",
    "PortfolioSnapshot",
    "PositionSnapshot",
    "Prediction",
    "Price",
    "Return",
    "Strategy",
    "Trade",
    "Universe",
    "UniverseMember",
]

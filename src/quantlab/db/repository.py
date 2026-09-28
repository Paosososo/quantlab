"""Data access helpers.

This is the only layer that knows about both SQLAlchemy and pandas.  Two
responsibilities justify its existence:

1.  **Idempotent upserts.**  Ingestion must be safe to re-run.  Every writer
    here uses ``INSERT ... ON CONFLICT DO UPDATE`` on the table's natural key,
    so re-running yesterday's load produces the same database rather than
    duplicate rows or an integrity error.

2.  **The Decimal/float boundary.**  Money is ``NUMERIC`` in the database and
    ``float64`` in pandas.  Converting in exactly one place means the rest of
    the codebase never has to think about ``Decimal`` arithmetic, and no other
    module can accidentally mix the two.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import Any, cast

import pandas as pd
from sqlalchemy import Select, Table, and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from quantlab.db import models as m
from quantlab.db.enums import AssetClass, ReturnDirection, ReturnMethod
from quantlab.exceptions import QuantlabError
from quantlab.logging import get_logger
from quantlab.timeutils import to_utc

log = get_logger(__name__)

PRICE_FLOAT_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


# ---------------------------------------------------------------------------
# Generic upsert
# ---------------------------------------------------------------------------
def upsert(
    session: Session,
    model: type[m.Base],
    rows: Sequence[dict[str, Any]],
    *,
    conflict_columns: Sequence[str],
    update_columns: Sequence[str] | None = None,
    chunk_size: int = 2_000,
) -> int:
    """Insert ``rows``, updating on conflict over ``conflict_columns``.

    Returns the number of rows submitted.  ``ON CONFLICT`` row counts are not
    portable across dialects, so we report what we sent rather than inventing a
    number we cannot verify.
    """
    if not rows:
        return 0

    dialect = session.get_bind().dialect.name
    # Typed as Any: the PostgreSQL and SQLite Insert constructs have the same
    # surface for our purposes but no common supertype that exposes
    # ``on_conflict_do_update``.
    insert_fn: Any
    if dialect == "postgresql":
        insert_fn = pg_insert
    elif dialect == "sqlite":
        insert_fn = sqlite_insert
    else:  # pragma: no cover - we only support these two
        raise QuantlabError("upsert not supported on this dialect", dialect=dialect)

    table = cast(Table, model.__table__)
    if update_columns is None:
        managed = set(conflict_columns) | {"id", "created_at"}
        update_columns = [c.name for c in table.columns if c.name not in managed]

    total = 0
    for start in range(0, len(rows), chunk_size):
        chunk = list(rows[start : start + chunk_size])
        stmt = insert_fn(table).values(chunk)
        if update_columns:
            stmt = stmt.on_conflict_do_update(
                index_elements=list(conflict_columns),
                set_={col: getattr(stmt.excluded, col) for col in update_columns},
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=list(conflict_columns))
        session.execute(stmt)
        total += len(chunk)
    return total


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
def get_asset_by_symbol(session: Session, symbol: str) -> m.Asset | None:
    return session.execute(select(m.Asset).where(m.Asset.symbol == symbol)).scalar_one_or_none()


def get_or_create_asset(
    session: Session,
    symbol: str,
    *,
    asset_class: AssetClass = AssetClass.EQUITY,
    name: str | None = None,
    exchange: str | None = None,
    currency: str = "USD",
    session_name: str = "us_equity",
    meta: dict[str, Any] | None = None,
) -> m.Asset:
    """Fetch an asset by symbol, creating it if it does not exist.

    The read-then-write is a time-of-check/time-of-use race: two Airflow tasks
    ingesting different symbols on the same day both call this for a shared
    benchmark ticker, both see nothing, and both INSERT.  ``uq_assets_symbol``
    turns the loser into an IntegrityError that would otherwise poison the whole
    transaction and fail an ingestion run for a reason unrelated to the data.

    The insert therefore runs inside a SAVEPOINT (``session.begin_nested``).  On
    conflict only the savepoint rolls back, the outer transaction survives, and
    we re-read the row the other writer committed.  This is the standard
    upsert-by-retry pattern and it relies on the unique constraint, not on
    locking, so it stays correct under any isolation level.
    """
    asset = get_asset_by_symbol(session, symbol)
    if asset is not None:
        return asset
    try:
        with session.begin_nested():
            asset = m.Asset(
                symbol=symbol,
                name=name,
                asset_class=asset_class,
                exchange=exchange,
                currency=currency.upper(),
                session_name=session_name,
                meta=meta,
            )
            session.add(asset)
            session.flush()
    except IntegrityError:
        # Another writer won the race.  Its row is authoritative.
        existing = get_asset_by_symbol(session, symbol)
        if existing is None:
            # The conflict was not the symbol collision we anticipated (a bad
            # currency, say).  Re-raising beats guessing.
            raise
        log.info("asset.create_raced", symbol=symbol, asset_id=existing.id)
        return existing
    log.info("asset.created", symbol=symbol, asset_id=asset.id)
    return asset


def list_assets(session: Session, *, asset_class: AssetClass | None = None) -> list[m.Asset]:
    stmt = select(m.Asset).order_by(m.Asset.symbol)
    if asset_class is not None:
        stmt = stmt.where(m.Asset.asset_class == asset_class)
    return list(session.execute(stmt).scalars())


def universe_symbols_as_of(session: Session, universe_code: str, as_of: dt.datetime) -> list[str]:
    """Point-in-time universe membership.

    This is the survivorship-bias guard.  A backtest that iterates over
    ``list_assets()`` is implicitly using today's membership for every
    historical date; this query is what it should call instead.
    """
    cutoff = to_utc(as_of)
    stmt = (
        select(m.Asset.symbol)
        .join(m.UniverseMember, m.UniverseMember.asset_id == m.Asset.id)
        .join(m.Universe, m.Universe.id == m.UniverseMember.universe_id)
        .where(
            m.Universe.code == universe_code,
            m.UniverseMember.valid_from <= cutoff,
            or_(m.UniverseMember.valid_to.is_(None), m.UniverseMember.valid_to > cutoff),
        )
        .order_by(m.Asset.symbol)
    )
    return list(session.execute(stmt).scalars())


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def upsert_prices(session: Session, rows: Sequence[dict[str, Any]]) -> int:
    return upsert(session, m.Price, rows, conflict_columns=("asset_id", "ts"))


def latest_price_ts(session: Session, asset_id: int) -> dt.datetime | None:
    """Watermark for incremental loading."""
    return session.execute(
        select(func.max(m.Price.ts)).where(m.Price.asset_id == asset_id)
    ).scalar_one_or_none()


def load_prices(
    session: Session,
    symbols: Iterable[str],
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    """Return a tidy price frame with columns
    ``[symbol, ts, available_at, open, high, low, close, adj_close, volume]``.

    ``as_of`` applies the availability predicate.  Passing it is how a caller
    asks "what could I have seen at this instant?"; omitting it returns
    everything and is only appropriate for reporting, never for feature
    construction.
    """
    symbols = list(symbols)
    stmt = (
        select(
            m.Asset.symbol,
            m.Price.ts,
            m.Price.available_at,
            m.Price.open,
            m.Price.high,
            m.Price.low,
            m.Price.close,
            m.Price.adj_close,
            m.Price.volume,
        )
        .join(m.Asset, m.Asset.id == m.Price.asset_id)
        .order_by(m.Asset.symbol, m.Price.ts)
    )
    if symbols:
        stmt = stmt.where(m.Asset.symbol.in_(symbols))
    if start is not None:
        stmt = stmt.where(m.Price.ts >= to_utc(start))
    if end is not None:
        stmt = stmt.where(m.Price.ts <= to_utc(end))
    if as_of is not None:
        stmt = stmt.where(m.Price.available_at <= to_utc(as_of))

    frame = _to_frame(session, stmt)
    return _coerce_price_frame(frame)


def _to_frame(session: Session, stmt: Select) -> pd.DataFrame:
    result = session.execute(stmt)
    return pd.DataFrame(result.mappings().all())


def _coerce_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "ts", "available_at", *PRICE_FLOAT_COLUMNS]).astype(
            dict.fromkeys(PRICE_FLOAT_COLUMNS, "float64")
        )
    out = frame.copy()
    for col in ("ts", "available_at"):
        out[col] = pd.to_datetime(out[col], utc=True)
    for col in PRICE_FLOAT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(_decimal_to_float).astype("float64")
    return out


def _decimal_to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def to_decimal(value: float | int | Decimal | None, places: int = 8) -> Decimal | None:
    """Float -> Decimal at the storage boundary, quantised to the column scale.

    Non-finite input becomes NULL, not ``Decimal('NaN')``.  PostgreSQL's NUMERIC
    accepts NaN, and a NaN silently satisfies every CHECK constraint because
    ``NaN >= 0`` evaluates to unknown rather than false.  A missing volume
    therefore used to be stored as a number that passes validation, compares
    false against every threshold, and poisons any aggregate computed over the
    column.  NULL is the honest representation of "not known".
    """
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    quant = Decimal(1).scaleb(-places)
    return Decimal(repr(numeric)).quantize(quant)


def wide_close_frame(prices: pd.DataFrame, field: str = "adj_close") -> pd.DataFrame:
    """Pivot a tidy price frame to ``ts`` x ``symbol``.

    Uses ``adj_close`` by default because returns must be computed on a
    split- and dividend-adjusted series; using raw ``close`` puts a fake -50%
    return on every split date.
    """
    if prices.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    wide = prices.pivot(index="ts", columns="symbol", values=field).sort_index()
    wide.index = pd.DatetimeIndex(wide.index).tz_convert("UTC")
    wide.columns.name = None
    return wide


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------
def upsert_returns(session: Session, rows: Sequence[dict[str, Any]]) -> int:
    return upsert(
        session,
        m.Return,
        rows,
        conflict_columns=("asset_id", "ts", "horizon_days", "direction", "method", "price_field"),
    )


def load_returns(
    session: Session,
    symbols: Iterable[str],
    *,
    horizon_days: int = 1,
    direction: ReturnDirection = ReturnDirection.TRAILING,
    method: ReturnMethod = ReturnMethod.SIMPLE,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    symbols = list(symbols)
    stmt = (
        select(m.Asset.symbol, m.Return.ts, m.Return.available_at, m.Return.value)
        .join(m.Asset, m.Asset.id == m.Return.asset_id)
        .where(
            m.Return.horizon_days == horizon_days,
            m.Return.direction == direction,
            m.Return.method == method,
        )
        .order_by(m.Asset.symbol, m.Return.ts)
    )
    if symbols:
        stmt = stmt.where(m.Asset.symbol.in_(symbols))
    if as_of is not None:
        stmt = stmt.where(m.Return.available_at <= to_utc(as_of))
    frame = _to_frame(session, stmt)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "ts", "available_at", "value"])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["value"] = frame["value"].astype("float64")
    return frame


# ---------------------------------------------------------------------------
# Economic data
# ---------------------------------------------------------------------------
def get_or_create_series(
    session: Session,
    code: str,
    *,
    frequency: str,
    title: str | None = None,
    units: str | None = None,
    publication_lag_days: int = 0,
    source: str = "fred",
) -> m.EconomicSeries:
    existing = session.execute(
        select(m.EconomicSeries).where(m.EconomicSeries.code == code)
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    series = m.EconomicSeries(
        code=code,
        title=title,
        units=units,
        frequency=frequency,
        publication_lag_days=publication_lag_days,
        source=source,
    )
    session.add(series)
    session.flush()
    return series


def upsert_economic(session: Session, rows: Sequence[dict[str, Any]]) -> int:
    return upsert(session, m.EconomicObservation, rows, conflict_columns=("series_id", "ts"))


def load_economic(
    session: Session,
    codes: Iterable[str],
    *,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    codes = list(codes)
    stmt = (
        select(
            m.EconomicSeries.code,
            m.EconomicObservation.ts,
            m.EconomicObservation.available_at,
            m.EconomicObservation.value,
        )
        .join(m.EconomicSeries, m.EconomicSeries.id == m.EconomicObservation.series_id)
        .order_by(m.EconomicSeries.code, m.EconomicObservation.ts)
    )
    if codes:
        stmt = stmt.where(m.EconomicSeries.code.in_(codes))
    if as_of is not None:
        stmt = stmt.where(m.EconomicObservation.available_at <= to_utc(as_of))
    frame = _to_frame(session, stmt)
    if frame.empty:
        return pd.DataFrame(columns=["code", "ts", "available_at", "value"])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    return frame


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def upsert_feature_values(session: Session, rows: Sequence[dict[str, Any]]) -> int:
    return upsert(
        session, m.FeatureValue, rows, conflict_columns=("feature_set_id", "asset_id", "name", "ts")
    )


def load_features(
    session: Session,
    feature_set_id: int,
    symbols: Iterable[str] | None = None,
    *,
    names: Sequence[str] | None = None,
    as_of: dt.datetime | None = None,
) -> pd.DataFrame:
    """Return features in long form: ``[symbol, name, ts, available_at, value]``."""
    stmt = (
        select(
            m.Asset.symbol,
            m.FeatureValue.name,
            m.FeatureValue.ts,
            m.FeatureValue.available_at,
            m.FeatureValue.value,
        )
        .join(m.Asset, m.Asset.id == m.FeatureValue.asset_id)
        .where(m.FeatureValue.feature_set_id == feature_set_id)
        .order_by(m.Asset.symbol, m.FeatureValue.ts, m.FeatureValue.name)
    )
    if symbols:
        stmt = stmt.where(m.Asset.symbol.in_(list(symbols)))
    if names:
        stmt = stmt.where(m.FeatureValue.name.in_(list(names)))
    if as_of is not None:
        stmt = stmt.where(m.FeatureValue.available_at <= to_utc(as_of))
    frame = _to_frame(session, stmt)
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "name", "ts", "available_at", "value"])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    return frame


def pivot_features(long_frame: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide: index ``(symbol, ts)``, one column per feature name."""
    if long_frame.empty:
        return pd.DataFrame()
    wide = long_frame.pivot_table(
        index=["symbol", "ts"], columns="name", values="value", aggfunc="last"
    )
    wide.columns.name = None
    return wide.sort_index()


# ---------------------------------------------------------------------------
# Backtest persistence
# ---------------------------------------------------------------------------
def bulk_insert(session: Session, model: type[m.Base], rows: Sequence[dict[str, Any]]) -> int:
    if not rows:
        return 0
    session.execute(cast(Table, model.__table__).insert(), list(rows))
    return len(rows)


def price_coverage(
    session: Session, symbol: str
) -> tuple[dt.datetime | None, dt.datetime | None, int]:
    """(first_ts, last_ts, row_count) for a symbol.  Used by health and validation."""
    stmt = (
        select(func.min(m.Price.ts), func.max(m.Price.ts), func.count(m.Price.id))
        .join(m.Asset, m.Asset.id == m.Price.asset_id)
        .where(m.Asset.symbol == symbol)
    )
    row = session.execute(stmt).one()
    return row[0], row[1], int(row[2] or 0)


def count_rows(session: Session, model: type[m.Base], *criteria: Any) -> int:
    stmt = select(func.count()).select_from(cast(Table, model.__table__))
    if criteria:
        stmt = stmt.where(and_(*criteria))
    return int(session.execute(stmt).scalar_one())


__all__ = [
    "PRICE_FLOAT_COLUMNS",
    "bulk_insert",
    "count_rows",
    "get_asset_by_symbol",
    "get_or_create_asset",
    "get_or_create_series",
    "latest_price_ts",
    "list_assets",
    "load_economic",
    "load_features",
    "load_prices",
    "load_returns",
    "pivot_features",
    "price_coverage",
    "to_decimal",
    "universe_symbols_as_of",
    "upsert",
    "upsert_economic",
    "upsert_feature_values",
    "upsert_prices",
    "upsert_returns",
    "wide_close_frame",
]

"""Declarative base and shared column conventions.

Two decisions worth defending in an interview live here.

1.  **Explicit constraint naming convention.**  Without it, Alembic autogenerate
    produces migrations that cannot drop unnamed constraints, and the same
    schema gets different constraint names on different databases.  The
    convention makes migrations deterministic.

2.  **Portable enums.**  ``Enum(..., native_enum=False)`` stores a VARCHAR with
    a CHECK constraint instead of a PostgreSQL ``CREATE TYPE``.  Native enums
    are painful to alter (you cannot remove a value without recreating the
    type) and do not exist on SQLite, which we use for fast unit tests.  The
    constraint still enforces the domain, so we lose nothing but the type name.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import DateTime, Dialect, MetaData, Numeric, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata_obj = MetaData(naming_convention=NAMING_CONVENTION)

#: Money, prices and share quantities.  ``NUMERIC`` rather than ``DOUBLE
#: PRECISION`` because the database is the system of record for accounting
#: quantities and exact decimal arithmetic is free there.  Statistical
#: quantities (returns, feature values, metrics) use double precision instead,
#: where the extra range matters more than the last decimal place.
Money = Numeric(20, 8)
Quantity = Numeric(24, 10)


class UTCDateTime(TypeDecorator[dt.datetime]):
    """A timestamp column that is always timezone-aware UTC in Python.

    PostgreSQL's ``TIMESTAMPTZ`` already round-trips an aware datetime, but
    SQLite has no timezone-aware type and hands back naive values.  Unit tests
    run on SQLite, so without this decorator every test would silently exercise
    naive timestamps while production used aware ones -- and the naive/aware
    boundary is precisely where timestamp-alignment bugs hide.

    The contract is enforced in both directions: writing a naive datetime is a
    programming error and raises, and reading always yields UTC.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: dt.datetime | None, dialect: Dialect) -> dt.datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime cannot be stored; attach a timezone before writing")
        as_utc = value.astimezone(dt.timezone.utc)
        # SQLite stores the string form and drops the offset, so hand it a
        # naive value that is already in UTC rather than a tagged one.
        return as_utc.replace(tzinfo=None) if dialect.name == "sqlite" else as_utc

    def process_result_value(
        self,
        value: dt.datetime | None,
        dialect: Dialect,  # noqa: ARG002 - required by the TypeDecorator interface
    ) -> dt.datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)


TZDateTime = UTCDateTime()


class Base(DeclarativeBase):
    metadata = metadata_obj

    def as_dict(self) -> dict[str, Any]:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pk = ", ".join(
            f"{column.name}={getattr(self, column.name)!r}" for column in self.__table__.primary_key
        )
        return f"<{type(self).__name__} {pk}>"


class TimestampMixin:
    """``created_at`` / ``updated_at`` maintained by the database."""

    created_at: Mapped[dt.datetime] = mapped_column(
        TZDateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        TZDateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


__all__ = [
    "NAMING_CONVENTION",
    "Base",
    "Money",
    "Quantity",
    "TZDateTime",
    "TimestampMixin",
    "UTCDateTime",
    "metadata_obj",
]

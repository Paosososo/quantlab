"""Shared FastAPI dependencies."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, Query
from sqlalchemy.orm import Session

from quantlab.db.session import get_sessionmaker
from quantlab.timeutils import UTC


def get_session() -> Iterator[Session]:
    """Request-scoped database session.

    Read-only endpoints still get a transaction, and it is rolled back rather
    than committed: an API that can accidentally write during a GET is an API
    that will eventually corrupt something.
    """
    factory = get_sessionmaker()
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


SessionDep = Depends(get_session)


@dataclass(frozen=True, slots=True)
class Pagination:
    limit: int
    offset: int


def pagination(
    limit: int = Query(default=500, ge=1, le=10_000, description="maximum rows to return"),
    offset: int = Query(default=0, ge=0),
) -> Pagination:
    return Pagination(limit=limit, offset=offset)


PaginationDep = Depends(pagination)


def as_of_parameter(
    as_of: dt.datetime | None = Query(
        default=None,
        description=(
            "Point-in-time cutoff.  When supplied, only observations that were "
            "published at or before this instant are returned, so the response is "
            "what a researcher standing at that moment could have seen.  This is "
            "the availability model described in docs/methodology.md."
        ),
    ),
) -> dt.datetime | None:
    if as_of is None:
        return None
    return as_of if as_of.tzinfo is not None else as_of.replace(tzinfo=UTC)


AsOfDep = Depends(as_of_parameter)


__all__ = [
    "AsOfDep",
    "Pagination",
    "PaginationDep",
    "SessionDep",
    "as_of_parameter",
    "get_session",
    "pagination",
]

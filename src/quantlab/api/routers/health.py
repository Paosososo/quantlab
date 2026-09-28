"""Liveness and readiness."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from quantlab import __version__
from quantlab.api.schemas import HealthResponse, ReadinessResponse
from quantlab.config import get_settings
from quantlab.db.session import get_engine, ping
from quantlab.timeutils import UTC

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness.  Answers even when the database is down, by design.

    A liveness probe that fails on a database outage causes the orchestrator to
    restart a perfectly healthy process, which does not fix the database and
    does lose in-flight requests.  Database state is reported here as a field,
    and gates readiness instead.
    """
    database_ok = ping()
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=get_settings().env,
        database="ok" if database_ok else "unavailable",
        checked_at=dt.datetime.now(tz=UTC),
    )


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(response: Response) -> ReadinessResponse:
    """Readiness.  Fails with 503 when the service cannot serve traffic."""
    if not ping():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(ready=False, database=False, detail="database unreachable")

    migrations_applied: bool | None = None
    detail: str | None = None
    try:
        with get_engine().connect() as connection:
            revision = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one_or_none()
        migrations_applied = revision is not None
        detail = f"alembic revision {revision}" if revision else "no alembic revision recorded"
    except Exception as exc:
        migrations_applied = False
        detail = f"could not read alembic_version: {type(exc).__name__}"

    if not migrations_applied:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        ready=bool(migrations_applied),
        database=True,
        migrations_applied=migrations_applied,
        detail=detail,
    )


__all__ = ["router"]

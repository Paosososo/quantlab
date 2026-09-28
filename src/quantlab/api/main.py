"""FastAPI application.

Read-mostly by design.  The API exposes what the pipelines produced and computes
a couple of analytics on demand; it does not kick off ingestion or training.
Long-running jobs triggered by an HTTP request need a task queue, a status
endpoint and a cancellation story, and Airflow already provides all three.  An
endpoint that blocks for four minutes training a model is a worse version of a
DAG.

The distinctive feature is ``as_of``.  Most read endpoints accept it, and when
supplied the response is what a researcher standing at that instant could have
seen.  That makes the availability model queryable from outside the system
rather than an internal convention.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from quantlab import __version__
from quantlab.api.errors import install_error_handlers
from quantlab.api.routers import analytics, backtests, features, health, market, models
from quantlab.config import get_settings
from quantlab.db.session import ping, reset_engine
from quantlab.logging import configure_logging, get_logger

log = get_logger(__name__)

DESCRIPTION = """
Read access to a financial research warehouse, plus on-demand portfolio analytics.

**Point-in-time queries.** Most endpoints accept an `as_of` parameter. Supplying
it filters to observations that were *published* at or before that instant, so
the response is the history as it stood then, not as it was later revised. This
is the availability model that the rest of the platform is built on; see
`docs/methodology.md`.

**Honest performance numbers.** `/backtests/{id}/performance` reports a deflated
Sharpe ratio that accounts for how many strategy configurations have been tried,
and the minimum track-record length needed for the result to be distinguishable
from zero.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    configure_logging()
    settings = get_settings()
    log.info(
        "api.starting",
        version=__version__,
        environment=settings.env,
        database_reachable=ping(),
    )
    yield
    reset_engine()
    log.info("api.stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="quantlab",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Permissive CORS only outside production: the dashboard runs on a different
    # port in development, and locking it down there costs time without buying
    # security on a localhost-only service.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.env != "production" else [],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        log.info(
            "api.request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round(elapsed_ms, 1),
        )
        return response

    install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(market.router)
    app.include_router(features.router)
    app.include_router(models.router)
    app.include_router(backtests.router)
    app.include_router(analytics.router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "name": "quantlab",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


__all__ = ["app", "create_app"]

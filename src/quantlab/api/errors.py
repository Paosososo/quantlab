"""Exception handling.

Every quantlab exception maps to an HTTP status and a consistent JSON body.  Two
rules:

*The exception type carries the meaning.*  A ``LookAheadError`` escaping into a
request is a 500 and is logged at error level, because it means the system did
something it promises never to do -- it is not a client mistake and must not be
reported as one.

*The response body never leaks internals.*  Context dictionaries from
``QuantlabError`` are structured and safe to return (symbols, dates, counts), but
tracebacks and connection strings are logged, not serialised.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from quantlab.exceptions import (
    DataQualityError,
    InsufficientDataError,
    LookAheadError,
    ModelError,
    QuantlabError,
    TemporalError,
)
from quantlab.logging import get_logger

log = get_logger(__name__)

#: Starlette renamed ``HTTP_422_UNPROCESSABLE_ENTITY`` to
#: ``HTTP_422_UNPROCESSABLE_CONTENT``.  Using the literal keeps the code working
#: on both versions and out of the deprecation warnings.
UNPROCESSABLE_CONTENT = 422


class NotFoundError(QuantlabError):
    """A requested resource does not exist."""


def _body(error: str, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": error, "message": message, "context": jsonable_encoder(context or {})}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_body("not_found", str(exc), exc.context),
        )

    @app.exception_handler(InsufficientDataError)
    async def _insufficient(request: Request, exc: InsufficientDataError) -> JSONResponse:  # noqa: ARG001
        # 422 rather than 400: the request is well formed, but the data the
        # server holds cannot satisfy it.
        return JSONResponse(
            status_code=UNPROCESSABLE_CONTENT,
            content=_body("insufficient_data", str(exc), exc.context),
        )

    @app.exception_handler(TemporalError)
    async def _temporal(request: Request, exc: TemporalError) -> JSONResponse:
        if isinstance(exc, LookAheadError):
            # Never a client error.  Something in the system tried to read the
            # future, which is a defect, and it is logged loudly.
            log.error("api.look_ahead_error", path=str(request.url.path), error=str(exc))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=_body(
                    "look_ahead_error",
                    "the server attempted an operation that would use future information",
                ),
            )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_body("temporal_error", str(exc), exc.context),
        )

    @app.exception_handler(DataQualityError)
    async def _data_quality(request: Request, exc: DataQualityError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            status_code=UNPROCESSABLE_CONTENT,
            content=_body("data_quality_error", str(exc), exc.context),
        )

    @app.exception_handler(ModelError)
    async def _model(request: Request, exc: ModelError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            status_code=UNPROCESSABLE_CONTENT,
            content=_body("model_error", str(exc), exc.context),
        )

    @app.exception_handler(QuantlabError)
    async def _generic(request: Request, exc: QuantlabError) -> JSONResponse:
        log.error("api.quantlab_error", path=str(request.url.path), error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_body("quantlab_error", str(exc), exc.context),
        )

    @app.exception_handler(SQLAlchemyError)
    async def _database(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        log.error("api.database_error", path=str(request.url.path), error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_body("database_error", "the database is unavailable or rejected the query"),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            status_code=UNPROCESSABLE_CONTENT,
            content=_body(
                "validation_error", "request validation failed", {"detail": exc.errors()}
            ),
        )


__all__ = ["NotFoundError", "install_error_handlers"]

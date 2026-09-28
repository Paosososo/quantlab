"""Exception hierarchy for quantlab.

Design rule enforced throughout the codebase: *never* swallow an exception
silently.  Either handle it and log at WARNING or above with context, or wrap it
in one of the types below and re-raise.  A bare ``except Exception: pass`` in a
research system silently converts a data bug into a plausible-looking backtest,
which is the worst failure mode this project can have.
"""

from __future__ import annotations

from typing import Any


class QuantlabError(Exception):
    """Root of the quantlab exception tree."""

    def __init__(self, message: str, **context: Any) -> None:
        self.context = context
        detail = " ".join(f"{k}={v!r}" for k, v in context.items())
        super().__init__(f"{message} ({detail})" if detail else message)


# --- configuration -------------------------------------------------------
class ConfigurationError(QuantlabError):
    """Settings are missing or mutually inconsistent."""


# --- ingestion -----------------------------------------------------------
class IngestionError(QuantlabError):
    """Base class for anything that goes wrong pulling external data."""


class ProviderError(IngestionError):
    """A data provider failed in a way specific to that provider."""


class RateLimitError(IngestionError):
    """The provider signalled that we are querying too fast.

    Carries ``retry_after`` in seconds when the provider tells us.
    """

    def __init__(self, message: str, retry_after: float | None = None, **context: Any) -> None:
        self.retry_after = retry_after
        super().__init__(message, retry_after=retry_after, **context)


class TransientHTTPError(IngestionError):
    """A retryable transport-level failure (5xx, timeout, connection reset)."""


class PermanentHTTPError(IngestionError):
    """A non-retryable response (4xx other than 429)."""


# --- data quality --------------------------------------------------------
class DataQualityError(QuantlabError):
    """A dataset violated a validation rule."""


class SchemaValidationError(DataQualityError):
    """Columns or dtypes did not match the declared schema."""


class DuplicateObservationError(DataQualityError):
    """The same (entity, timestamp) key appeared more than once."""


# --- temporal correctness -------------------------------------------------
class TemporalError(QuantlabError):
    """Base class for anything involving time ordering."""


class LookAheadError(TemporalError):
    """Code attempted to read information that was not available at the as-of time.

    This is deliberately loud.  Any occurrence is a bug that would inflate
    backtest performance, so it must fail the run rather than warn.
    """


class NonMonotonicTimestampError(TemporalError):
    """A time series was not sorted strictly increasing on its time index."""


# --- modelling -----------------------------------------------------------
class ModelError(QuantlabError):
    """Base class for model lifecycle problems."""


class NotFittedError(ModelError):
    """predict() was called before fit()."""


class InsufficientDataError(ModelError):
    """Not enough observations to fit or evaluate."""


# --- backtesting ---------------------------------------------------------
class BacktestError(QuantlabError):
    """Base class for backtest engine failures."""


class InsufficientCashError(BacktestError):
    """An order would push cash below the allowed floor."""


class InvalidOrderError(BacktestError):
    """An order was malformed (non-finite quantity, unknown symbol, zero size)."""


__all__ = [
    "BacktestError",
    "ConfigurationError",
    "DataQualityError",
    "DuplicateObservationError",
    "IngestionError",
    "InsufficientCashError",
    "InsufficientDataError",
    "InvalidOrderError",
    "LookAheadError",
    "ModelError",
    "NonMonotonicTimestampError",
    "NotFittedError",
    "PermanentHTTPError",
    "ProviderError",
    "QuantlabError",
    "RateLimitError",
    "SchemaValidationError",
    "TemporalError",
    "TransientHTTPError",
]

"""Timestamp normalisation and the availability model.

This module encodes the project's single most important idea, so it is worth
reading before anything else.

Two distinct timestamps
-----------------------
Every observation in this system carries two timestamps:

``ts``
    The instant the observation *describes*.  For a daily price bar this is the
    session date.  For a macroeconomic series it is the period the number
    refers to (e.g. ``2024-01-01`` for January CPI).

``available_at``
    The earliest instant at which a researcher standing in real time could have
    known the value.  For a price bar that is the session close.  For a macro
    series it is the publication instant, which can be weeks after ``ts``.

Nearly every look-ahead bug in quantitative research is the result of treating
these two as the same thing.  Storing them separately turns "did I peek at the
future?" from a code-review question into a SQL predicate:
``WHERE available_at <= :decision_time``.

Timezone policy
---------------
Everything inside the system is timezone-aware UTC.  Naive datetimes are
rejected at the boundary rather than coerced, because silent coercion of a
naive local timestamp is itself a source of misalignment.  Exchange-local
session semantics live in :class:`SessionSpec`, which is the only place that
knows "the US equity session closes at 16:00 America/New_York".
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from quantlab.exceptions import NonMonotonicTimestampError, TemporalError

UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Session specifications
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SessionSpec:
    """How a market's session date maps onto a wall-clock availability instant.

    ``close_local`` is the regular-hours close in ``tz``.  We deliberately do
    *not* model early closes (half-days).  That is a documented limitation: it
    makes ``available_at`` a few hours late on a handful of days per year, which
    is conservative -- it can never make data available earlier than reality.
    Erring conservative is the correct direction for a bias-prevention system.
    """

    name: str
    tz: str
    close_local: dt.time

    def close_instant(self, session_date: dt.date) -> dt.datetime:
        """UTC instant at which the session's data becomes knowable."""
        local = dt.datetime.combine(session_date, self.close_local, tzinfo=ZoneInfo(self.tz))
        return local.astimezone(UTC)


US_EQUITY_SESSION = SessionSpec("us_equity", "America/New_York", dt.time(16, 0))
LSE_SESSION = SessionSpec("lse", "Europe/London", dt.time(16, 30))
TSE_SESSION = SessionSpec("tse", "Asia/Tokyo", dt.time(15, 0))
SET_SESSION = SessionSpec("set", "Asia/Bangkok", dt.time(16, 30))

SESSIONS: dict[str, SessionSpec] = {
    s.name: s for s in (US_EQUITY_SESSION, LSE_SESSION, TSE_SESSION, SET_SESSION)
}


def get_session(name: str) -> SessionSpec:
    try:
        return SESSIONS[name]
    except KeyError as exc:
        raise TemporalError("unknown session", name=name, known=sorted(SESSIONS)) from exc


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def to_utc(value: dt.datetime | dt.date | str | pd.Timestamp) -> dt.datetime:
    """Normalise a single timestamp-like value to a timezone-aware UTC datetime.

    A ``date`` is interpreted as UTC midnight.  A naive ``datetime`` is
    rejected: guessing its zone is exactly the kind of silent assumption that
    misaligns series across markets.
    """
    if isinstance(value, str):
        value = pd.Timestamp(value).to_pydatetime()
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            raise TemporalError(
                "naive datetime rejected; attach a timezone before entering the system",
                value=value.isoformat(),
            )
        return value.astimezone(UTC)
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day, tzinfo=UTC)
    raise TemporalError("unsupported timestamp type", type=type(value).__name__)


def session_date_to_ts(session_date: dt.date) -> dt.datetime:
    """Canonical ``ts`` for a daily bar: UTC midnight of the session date.

    Using midnight (rather than the close) as ``ts`` keeps the *identity* of a
    bar tied to its calendar date, which is what joins across data sources need,
    while ``available_at`` carries the knowability semantics.
    """
    return dt.datetime(session_date.year, session_date.month, session_date.day, tzinfo=UTC)


def normalise_index(
    frame: pd.DataFrame,
    column: str,
    *,
    sort: bool = True,
    require_unique: bool = True,
) -> pd.DataFrame:
    """Coerce ``column`` to UTC datetimes, optionally sorting and checking uniqueness."""
    out = frame.copy()
    series = pd.to_datetime(out[column], utc=True, errors="raise")
    out[column] = series
    if sort:
        out = out.sort_values(column, kind="mergesort").reset_index(drop=True)
    if require_unique and out[column].duplicated().any():
        dupes = out.loc[out[column].duplicated(keep=False), column].unique()[:5]
        raise NonMonotonicTimestampError(
            "duplicate timestamps after normalisation",
            column=column,
            examples=[str(d) for d in dupes],
        )
    return out


def assert_monotonic(index: pd.Index, *, strict: bool = True, label: str = "index") -> None:
    """Raise if ``index`` is not increasing.  Used as a precondition everywhere."""
    if len(index) <= 1:
        return
    increasing = index.is_monotonic_increasing
    if not increasing:
        raise NonMonotonicTimestampError("index is not increasing", label=label)
    if strict and index.has_duplicates:
        raise NonMonotonicTimestampError("index has duplicate values", label=label)


def ensure_utc_index(frame: pd.DataFrame, *, label: str = "frame") -> pd.DataFrame:
    """Validate that ``frame`` has a sorted, unique, tz-aware UTC DatetimeIndex."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TemporalError("expected a DatetimeIndex", label=label, got=type(frame.index).__name__)
    if frame.index.tz is None:
        raise TemporalError("expected a timezone-aware index", label=label)
    idx = frame.index
    if str(idx.tz) != "UTC":
        frame = frame.tz_convert("UTC")
    assert_monotonic(frame.index, strict=True, label=label)
    return frame


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def price_available_at(
    session_date: dt.date, session: SessionSpec = US_EQUITY_SESSION
) -> dt.datetime:
    """When a daily bar for ``session_date`` becomes knowable."""
    return session.close_instant(session_date)


def macro_available_at(
    observation_date: dt.date,
    publication_lag_days: int,
    *,
    publication_time_utc: dt.time = dt.time(13, 30),
) -> dt.datetime:
    """When a macro observation becomes knowable, given a publication lag.

    The free FRED CSV endpoint returns the *latest* vintage of a series, not the
    value as first published, so we cannot recover the true release calendar
    from it.  Applying a per-series lag is an approximation.  It is documented
    as a limitation in ``docs/methodology.md`` and the lag is deliberately set
    to the conservative (long) end of each series' historical release delay so
    that features never see a number earlier than reality.
    """
    if publication_lag_days < 0:
        raise TemporalError("publication lag must be non-negative", lag=publication_lag_days)
    released = observation_date + dt.timedelta(days=publication_lag_days)
    return dt.datetime.combine(released, publication_time_utc, tzinfo=UTC)


def as_of_filter(
    frame: pd.DataFrame, as_of: dt.datetime, *, column: str = "available_at"
) -> pd.DataFrame:
    """Return only the rows knowable at ``as_of``.

    This is the runtime enforcement of the availability model.  Any code path
    that builds features or signals must pass through here (or the equivalent
    SQL predicate) rather than slicing on ``ts``.
    """
    if column not in frame.columns:
        raise TemporalError("frame has no availability column", column=column)
    cutoff = pd.Timestamp(to_utc(as_of))
    available = pd.to_datetime(frame[column], utc=True)
    return frame.loc[available <= cutoff]


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------
def annualisation_factor(periods_per_year: int) -> float:
    """sqrt(periods_per_year), the scaling used for volatility and Sharpe."""
    if periods_per_year <= 0:
        raise TemporalError("periods_per_year must be positive", value=periods_per_year)
    return float(periods_per_year) ** 0.5


def infer_periods_per_year(index: pd.DatetimeIndex, *, default: int = 252) -> int:
    """Infer observation frequency from spacing, falling back to ``default``.

    Uses the median gap rather than the mean so that a single long gap (a data
    outage, a market holiday cluster) does not distort the estimate.
    """
    if len(index) < 3:
        return default
    deltas = pd.Series(index).diff().dropna().dt.total_seconds()
    median_seconds = float(deltas.median())
    if median_seconds <= 0:
        return default
    seconds_per_year = 365.25 * 24 * 3600
    if median_seconds <= 3600 * 6:  # intraday
        return max(1, int(round(seconds_per_year / median_seconds)))
    if median_seconds <= 3 * 24 * 3600:  # daily-ish, incl. weekend gaps
        return default
    if median_seconds <= 10 * 24 * 3600:
        return 52
    if median_seconds <= 45 * 24 * 3600:
        return 12
    if median_seconds <= 130 * 24 * 3600:
        return 4
    return 1


__all__ = [
    "LSE_SESSION",
    "SESSIONS",
    "SET_SESSION",
    "TSE_SESSION",
    "US_EQUITY_SESSION",
    "UTC",
    "SessionSpec",
    "annualisation_factor",
    "as_of_filter",
    "assert_monotonic",
    "ensure_utc_index",
    "get_session",
    "infer_periods_per_year",
    "macro_available_at",
    "normalise_index",
    "price_available_at",
    "session_date_to_ts",
    "to_utc",
]

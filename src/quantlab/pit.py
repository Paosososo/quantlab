"""Point-in-time views.

The problem
-----------
Look-ahead bias is not usually introduced deliberately.  It arrives through an
innocuous line like ``df.loc[:, "close"].rolling(20).mean()`` computed on the
whole history and then joined back onto a signal, or a ``fillna(method="bfill")``
that pulls tomorrow's value into today's gap.  Reviewing every line for this is
unreliable.

The approach
------------
Make the future *unavailable* rather than merely discouraged.  A strategy or
feature function is handed a :class:`PointInTimeFrame`, which physically holds
only the rows whose ``available_at`` is at or before the as-of instant.  There
is no attribute on it that returns the full history.  Code that wants the future
has to go around the object, which is visible in review, instead of slipping
through inside a pandas one-liner.

This is a *structural* control.  It is complemented by a *behavioural* test
(:mod:`quantlab.features.guards`) that perturbs future data and asserts past
feature values do not move.  Neither alone is sufficient: the view stops the
common accident, the test catches the clever one.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantlab.exceptions import LookAheadError
from quantlab.timeutils import to_utc


class PointInTimeFrame:
    """A read-only window onto a time-indexed frame, truncated at ``as_of``.

    Parameters
    ----------
    frame:
        Must contain ``time_column`` and ``availability_column``.
    as_of:
        The decision instant.  Only rows knowable at or before this are kept.
    """

    __slots__ = ("_as_of", "_availability_column", "_time_column", "_visible")

    @classmethod
    def from_visible_prefix(
        cls,
        visible: pd.DataFrame,
        as_of: dt.datetime,
        *,
        time_column: str = "ts",
        availability_column: str = "available_at",
        availability_is_sorted: bool = False,
    ) -> PointInTimeFrame:
        """Build a view from rows a caller has *already* proven to be visible.

        The general constructor filters, sorts and copies, which is O(n) per
        call.  A backtest builds one view per bar, so that is O(n^2) over a run
        and it dominates the whole engine.
        :class:`quantlab.backtesting.market.MarketData` keeps each symbol sorted
        once and locates the cut point with a binary search, then calls this.

        The safety property is preserved exactly.  The object still physically
        holds only the visible rows, and the invariant is *re-checked* here
        rather than trusted, so a caller that miscomputes the prefix gets a
        LookAheadError instead of a silent leak.  Speed is bought by avoiding
        repeated work, never by weakening the guarantee.
        """
        instance = cls.__new__(cls)
        instance._as_of = to_utc(as_of)
        instance._time_column = time_column
        instance._availability_column = availability_column
        instance._visible = visible
        if not visible.empty:
            # When the caller guarantees the column is non-decreasing (MarketData
            # verifies this once, at construction) the newest row is the last
            # one, so the check is O(1) instead of O(k).  Without that guarantee
            # we scan, because an unchecked prefix is exactly the bug this class
            # exists to prevent.
            column = visible[availability_column]
            newest = (
                pd.Timestamp(column.iloc[-1])
                if availability_is_sorted
                else pd.to_datetime(column, utc=True).max()
            )
            if pd.Timestamp(newest).tz is None:
                newest = pd.Timestamp(newest).tz_localize("UTC")
            if newest > pd.Timestamp(instance._as_of):
                raise LookAheadError(
                    "prefix contains a row that was not yet available",
                    newest_available=str(newest),
                    as_of=instance._as_of.isoformat(),
                )
        return instance

    def __init__(
        self,
        frame: pd.DataFrame,
        as_of: dt.datetime,
        *,
        time_column: str = "ts",
        availability_column: str = "available_at",
    ) -> None:
        if time_column not in frame.columns:
            raise LookAheadError("frame lacks a time column", column=time_column)
        if availability_column not in frame.columns:
            raise LookAheadError(
                "frame lacks an availability column; refusing to guess it",
                column=availability_column,
            )
        self._as_of = to_utc(as_of)
        self._time_column = time_column
        self._availability_column = availability_column

        available = pd.to_datetime(frame[availability_column], utc=True)
        mask = available <= pd.Timestamp(self._as_of)
        # ``.copy()`` matters: without it the caller could reach the parent
        # frame through the view's ``_mgr`` and see hidden rows.
        self._visible = frame.loc[mask].sort_values(time_column).reset_index(drop=True).copy()

    # -- properties --------------------------------------------------------
    @property
    def as_of(self) -> dt.datetime:
        return self._as_of

    @property
    def columns(self) -> pd.Index:
        return self._visible.columns

    def __len__(self) -> int:
        return len(self._visible)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<PointInTimeFrame as_of={self._as_of.isoformat()} rows={len(self._visible)}>"

    # -- access ------------------------------------------------------------
    def frame(self) -> pd.DataFrame:
        """A copy of everything visible at ``as_of``."""
        return self._visible.copy()

    def tail(self, n: int) -> pd.DataFrame:
        """The most recent ``n`` visible rows."""
        if n <= 0:
            raise ValueError("n must be positive")
        return self._visible.tail(n).copy()

    def values(self, column: str, n: int | None = None) -> np.ndarray:
        """Raw numpy values for one visible column, optionally the last ``n``.

        The cheap accessor.  :meth:`series` builds an indexed pandas Series,
        which costs more than the arithmetic most strategies then do with it;
        over a few thousand bars that overhead dominates a backtest.  Strategies
        that only need a rolling window should call this.
        """
        if column not in self._visible.columns:
            raise KeyError(column)
        array = self._visible[column].to_numpy(dtype="float64")
        return array if n is None else array[-n:]

    def series(self, column: str) -> pd.Series:
        """One visible column, indexed by time."""
        if column not in self._visible.columns:
            raise KeyError(column)
        out = self._visible.set_index(self._time_column)[column]
        out.name = column
        return out.copy()

    def latest(self, column: str | None = None) -> pd.Series | float | None:
        """The most recent visible row, or one field of it."""
        if self._visible.empty:
            return None
        row = self._visible.iloc[-1]
        if column is None:
            return row.copy()
        return float(row[column])

    def value_at(self, ts: dt.datetime, column: str) -> float | None:
        """Value at an exact timestamp, if it is visible."""
        target = pd.Timestamp(to_utc(ts))
        if target > pd.Timestamp(self._as_of):
            raise LookAheadError(
                "requested a timestamp after the as-of instant",
                requested=target.isoformat(),
                as_of=self._as_of.isoformat(),
            )
        hit = self._visible.loc[self._visible[self._time_column] == target]
        if hit.empty:
            return None
        return float(hit.iloc[-1][column])

    def require_history(self, minimum: int, *, label: str = "series") -> pd.DataFrame:
        """Return the visible frame, raising when there is not enough of it.

        Used by features with a lookback window so that a short warm-up period
        produces an explicit error instead of a silently wrong number.
        """
        if len(self._visible) < minimum:
            raise LookAheadError(
                "insufficient visible history at as-of time",
                label=label,
                have=len(self._visible),
                need=minimum,
                as_of=self._as_of.isoformat(),
            )
        return self._visible.copy()


class PointInTimeUniverse:
    """A collection of per-symbol point-in-time frames sharing one as-of."""

    __slots__ = ("_as_of", "_frames")

    def __init__(self, frames: dict[str, PointInTimeFrame], as_of: dt.datetime) -> None:
        self._frames = frames
        self._as_of = to_utc(as_of)

    @classmethod
    def from_tidy(
        cls,
        tidy: pd.DataFrame,
        as_of: dt.datetime,
        *,
        symbol_column: str = "symbol",
        time_column: str = "ts",
        availability_column: str = "available_at",
    ) -> PointInTimeUniverse:
        frames = {
            str(symbol): PointInTimeFrame(
                group,
                as_of,
                time_column=time_column,
                availability_column=availability_column,
            )
            for symbol, group in tidy.groupby(symbol_column, sort=True)
        }
        return cls(frames, as_of)

    @property
    def as_of(self) -> dt.datetime:
        return self._as_of

    @property
    def symbols(self) -> Sequence[str]:
        return sorted(self._frames)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._frames

    def __getitem__(self, symbol: str) -> PointInTimeFrame:
        try:
            return self._frames[symbol]
        except KeyError as exc:
            raise KeyError(f"symbol {symbol!r} is not in this universe") from exc

    def get(self, symbol: str) -> PointInTimeFrame | None:
        return self._frames.get(symbol)


__all__ = ["PointInTimeFrame", "PointInTimeUniverse"]

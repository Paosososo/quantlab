"""In-memory market data for the engine.

Holds one sorted frame per symbol plus a merged timeline.  The engine walks the
timeline; on each date it can ask for that date's bar for any symbol, and it can
build a point-in-time view of everything visible at that date's close.

Symbols with different calendars (a US ETF and a London one, say) are handled by
the union timeline: a symbol simply has no bar on a date it did not trade, and
the engine skips it rather than forward-filling.  Forward-filling prices across a
market holiday would invent trades at prices that never existed.

Performance note (and why it is a correctness matter too)
---------------------------------------------------------
The obvious implementation of "everything visible at time t" is a boolean filter
over the whole panel, run once per bar.  That is O(n) per bar and O(n^2) over a
run; on a three-symbol, fifteen-year panel it made a single backtest take
minutes, which in practice means researchers stop running the safe path and
start hand-rolling vectorised shortcuts -- the exact behaviour this project
exists to prevent.

The fix keeps the guarantee and drops the cost.  Availability is monotone within
a symbol (later sessions are published later), so the visible set at any instant
is a *prefix* of that symbol's sorted rows, and its length is one binary search.
Slicing a prefix costs O(k) instead of scanning O(n), and
:meth:`PointInTimeFrame.from_visible_prefix` re-validates the boundary rather
than trusting it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from quantlab.backtesting.types import Bar
from quantlab.exceptions import BacktestError
from quantlab.pit import PointInTimeFrame, PointInTimeUniverse
from quantlab.timeutils import to_utc

REQUIRED_COLUMNS = ("symbol", "ts", "available_at", "open", "high", "low", "close")
_PRICE_FIELDS = ("open", "high", "low", "close", "adj_close", "volume")


class MarketData:
    """Sorted price history for a set of symbols."""

    __slots__ = (
        "_avail_ns",
        "_by_symbol",
        "_fields",
        "_log_returns",
        "_positions",
        "_tidy",
        "_timeline",
        "_ts_ns",
    )

    def __init__(self, tidy: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in tidy.columns]
        if missing:
            raise BacktestError("market data is missing columns", missing=missing)
        frame = tidy.copy()
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
        if "adj_close" not in frame.columns:
            frame["adj_close"] = frame["close"]
        if "volume" not in frame.columns:
            frame["volume"] = float("nan")
        frame = frame.sort_values(["symbol", "ts"]).reset_index(drop=True)

        if frame.duplicated(subset=["symbol", "ts"]).any():
            raise BacktestError("market data contains duplicate (symbol, ts) rows")

        self._tidy = frame
        self._by_symbol: dict[str, pd.DataFrame] = {}
        self._ts_ns: dict[str, np.ndarray] = {}
        self._avail_ns: dict[str, np.ndarray] = {}
        self._positions: dict[str, dict[int, int]] = {}
        self._fields: dict[str, dict[str, np.ndarray]] = {}
        self._log_returns: dict[str, dict[str, np.ndarray]] = {}

        for symbol, group in frame.groupby("symbol", sort=True):
            key = str(symbol)
            ordered = group.reset_index(drop=True)
            self._by_symbol[key] = ordered
            ts_values = ordered["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
            avail_values = ordered["available_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
            # Availability must be non-decreasing for the prefix trick to be
            # valid.  It is by construction (session closes advance with session
            # dates), but a bad loader could break it, so check rather than
            # assume: an unsorted availability column would silently hide rows.
            if np.any(np.diff(avail_values) < 0):
                raise BacktestError(
                    "availability is not non-decreasing within a symbol", symbol=key
                )
            self._ts_ns[key] = ts_values
            self._avail_ns[key] = avail_values
            self._positions[key] = {int(v): i for i, v in enumerate(ts_values)}
            self._fields[key] = {
                field: ordered[field].to_numpy(dtype="float64")
                for field in _PRICE_FIELDS
                if field in ordered.columns
            }
            self._log_returns[key] = {}

        self._timeline = pd.DatetimeIndex(np.sort(frame["ts"].unique()))

    # -- construction ------------------------------------------------------
    @classmethod
    def from_frames(cls, frames: Sequence[pd.DataFrame]) -> MarketData:
        return cls(pd.concat(list(frames), ignore_index=True))

    # -- properties --------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        return sorted(self._by_symbol)

    @property
    def timeline(self) -> pd.DatetimeIndex:
        return self._timeline

    @property
    def tidy(self) -> pd.DataFrame:
        return self._tidy

    def __len__(self) -> int:
        return len(self._timeline)

    # -- access ------------------------------------------------------------
    @staticmethod
    def _as_ns(ts: dt.datetime | pd.Timestamp) -> int:
        return int(pd.Timestamp(to_utc(ts)).value)

    def has(self, symbol: str, ts: dt.datetime) -> bool:
        positions = self._positions.get(symbol)
        return positions is not None and self._as_ns(ts) in positions

    def bar(self, symbol: str, ts: dt.datetime) -> Bar | None:
        positions = self._positions.get(symbol)
        if positions is None:
            return None
        index = positions.get(self._as_ns(ts))
        if index is None:
            return None
        fields = self._fields[symbol]
        volume = fields["volume"][index] if "volume" in fields else float("nan")
        return Bar(
            symbol=symbol,
            ts=pd.Timestamp(self._ts_ns[symbol][index], tz="UTC").to_pydatetime(),
            open=float(fields["open"][index]),
            high=float(fields["high"][index]),
            low=float(fields["low"][index]),
            close=float(fields["close"][index]),
            adj_close=float(fields["adj_close"][index]),
            volume=float(volume),
            available_at=pd.Timestamp(self._avail_ns[symbol][index], tz="UTC").to_pydatetime(),
        )

    def bars_at(self, ts: dt.datetime) -> dict[str, Bar]:
        out: dict[str, Bar] = {}
        for symbol in self._by_symbol:
            bar = self.bar(symbol, ts)
            if bar is not None:
                out[symbol] = bar
        return out

    def closes_at(self, ts: dt.datetime, field: str = "close") -> dict[str, float]:
        return {s: b.price(field) for s, b in self.bars_at(ts).items()}

    def last_known_closes(self, ts: dt.datetime, field: str = "close") -> dict[str, float]:
        """Most recent close at or before ``ts`` for every symbol.

        Needed to mark a position on a date its own market was closed.  Uses the
        last *actual* observation rather than interpolating.
        """
        key = self._as_ns(ts)
        out: dict[str, float] = {}
        for symbol, ts_values in self._ts_ns.items():
            cut = int(np.searchsorted(ts_values, key, side="right"))
            if cut > 0:
                out[symbol] = float(self._fields[symbol][field][cut - 1])
        return out

    def visible_count(self, symbol: str, as_of: dt.datetime) -> int:
        """How many of ``symbol``'s rows were published at or before ``as_of``."""
        avail = self._avail_ns.get(symbol)
        if avail is None:
            return 0
        return int(np.searchsorted(avail, self._as_ns(as_of), side="right"))

    def point_in_time(self, as_of: dt.datetime) -> PointInTimeUniverse:
        """Everything knowable at ``as_of``, as a leak-proof view.

        Built from per-symbol prefixes rather than a whole-panel filter; see the
        module docstring for why that matters.
        """
        cutoff = self._as_ns(as_of)
        frames: dict[str, PointInTimeFrame] = {}
        for symbol, avail in self._avail_ns.items():
            visible = int(np.searchsorted(avail, cutoff, side="right"))
            if visible == 0:
                continue
            frames[symbol] = PointInTimeFrame.from_visible_prefix(
                self._by_symbol[symbol].iloc[:visible],
                as_of,
                availability_is_sorted=True,
            )
        return PointInTimeUniverse(frames, to_utc(as_of))

    def trailing_log_returns(
        self,
        symbol: str,
        ts: dt.datetime,
        *,
        lookback: int,
        field: str = "adj_close",
        inclusive: bool = True,
    ) -> np.ndarray:
        """Log returns over the ``lookback`` bars ending at (or before) ``ts``.

        ``inclusive=False`` excludes ``ts``'s own bar.  That matters for the
        volatility estimate used to size an order being *filled* at ``ts``:
        including that bar would be a one-day look-ahead inside the risk model,
        which is subtle enough to be easy to miss.
        """
        ts_values = self._ts_ns.get(symbol)
        if ts_values is None:
            return np.empty(0, dtype="float64")
        side: Literal["left", "right"] = "right" if inclusive else "left"
        cut = int(np.searchsorted(ts_values, self._as_ns(ts), side=side))
        if cut < 2:
            return np.empty(0, dtype="float64")

        cached = self._log_returns[symbol].get(field)
        if cached is None:
            prices = self._fields[symbol][field]
            with np.errstate(divide="ignore", invalid="ignore"):
                logs = np.log(np.where(prices > 0, prices, np.nan))
            cached = np.diff(logs, prepend=np.nan)
            self._log_returns[symbol][field] = cached

        window = cached[max(1, cut - lookback) : cut]
        return window[np.isfinite(window)]

    def trailing_volatility(
        self,
        symbol: str,
        ts: dt.datetime,
        *,
        lookback: int,
        field: str = "adj_close",
        inclusive: bool = True,
        annualise: bool = False,
        periods_per_year: int = 252,
        minimum: int = 20,
    ) -> float | None:
        returns = self.trailing_log_returns(
            symbol, ts, lookback=lookback, field=field, inclusive=inclusive
        )
        if returns.size < minimum:
            return None
        value = float(np.std(returns, ddof=1))
        if not np.isfinite(value) or value <= 0:
            return None
        return value * float(np.sqrt(periods_per_year)) if annualise else value

    def iter_timeline(
        self, start: dt.datetime | None = None, end: dt.datetime | None = None
    ) -> Iterator[pd.Timestamp]:
        idx = self._timeline
        if start is not None:
            idx = idx[idx >= pd.Timestamp(to_utc(start))]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(to_utc(end))]
        yield from idx


__all__ = ["REQUIRED_COLUMNS", "MarketData"]

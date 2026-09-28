"""Strategy interface and a few reference strategies.

A strategy sees a :class:`DecisionContext` and returns target portfolio weights.
It cannot see the future, because everything reachable from the context is a
point-in-time view built at the decision instant.  It also cannot place an order
directly, so it cannot choose its own execution price; the engine owns that.

Returning weights rather than orders is deliberate.  Weights are scale-free, so
the same strategy runs on any portfolio size, and the sizing and execution
policy can be varied without touching the alpha logic.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtesting.types import TargetWeight
from quantlab.exceptions import BacktestError
from quantlab.pit import PointInTimeFrame, PointInTimeUniverse


@dataclass(frozen=True, slots=True)
class PortfolioView:
    """Read-only snapshot of the portfolio handed to a strategy."""

    equity: float
    cash: float
    weights: Mapping[str, float]
    quantities: Mapping[str, float]

    def weight(self, symbol: str) -> float:
        return float(self.weights.get(symbol, 0.0))


class PredictionBook:
    """Model forecasts, indexed for O(1) point-in-time lookup.

    The straightforward implementation filters a predictions DataFrame on every
    bar for every symbol, which is O(rows) per lookup and turns a backtest into
    an O(n^2) job.  This builds the index once.

    The availability guarantee is unchanged and is the reason the value and its
    availability are stored together: a lookup returns the forecast only when it
    had been produced by the as-of instant.  A prediction with a missing or
    future ``available_at`` is simply invisible.
    """

    __slots__ = ("_by_key", "_columns")

    REQUIRED = ("symbol", "ts", "available_at", "y_pred")

    def __init__(self, predictions: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED if c not in predictions.columns]
        if missing:
            raise BacktestError(
                "predictions must carry an 'available_at' column so the engine "
                "can prove each forecast existed at decision time",
                required=list(self.REQUIRED),
                missing=missing,
            )
        frame = predictions.copy()
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
        self._columns = list(frame.columns)
        self._by_key: dict[tuple[str, int], tuple[int, float]] = {}
        for symbol, ts, available_at, y_pred in zip(
            frame["symbol"].astype(str),
            frame["ts"].to_numpy(dtype="datetime64[ns]").astype("int64"),
            frame["available_at"].to_numpy(dtype="datetime64[ns]").astype("int64"),
            frame["y_pred"].to_numpy(dtype="float64"),
            strict=True,
        ):
            # Later rows win, matching the "corrections arrive later" convention
            # used throughout the ingestion layer.
            self._by_key[(symbol, int(ts))] = (int(available_at), float(y_pred))

    def __len__(self) -> int:
        return len(self._by_key)

    def lookup(self, symbol: str, ts: dt.datetime, as_of: dt.datetime) -> float | None:
        entry = self._by_key.get((symbol, int(pd.Timestamp(ts).value)))
        if entry is None:
            return None
        available_ns, value = entry
        if available_ns > int(pd.Timestamp(as_of).value):
            return None
        return value


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Everything a strategy is allowed to know at one decision instant."""

    ts: dt.datetime
    as_of: dt.datetime
    prices: PointInTimeUniverse
    portfolio: PortfolioView
    #: Optional per-symbol feature frames, already truncated at ``as_of``.
    features: Mapping[str, PointInTimeFrame] = field(default_factory=dict)
    #: Optional model predictions, filtered by availability at lookup time.
    predictions: PredictionBook | None = None

    def close_series(self, symbol: str, field_name: str = "adj_close") -> pd.Series:
        """Full visible history as a pandas Series.

        Convenient but not cheap.  Prefer :meth:`recent` inside a strategy that
        only needs a rolling window.
        """
        frame = self.prices.get(symbol)
        if frame is None or len(frame) == 0:
            return pd.Series(dtype="float64")
        return frame.series(field_name)

    def recent(self, symbol: str, n: int, field_name: str = "adj_close") -> np.ndarray:
        """The last ``n`` visible values of ``field_name``, as a numpy array."""
        frame = self.prices.get(symbol)
        if frame is None or len(frame) == 0:
            return np.empty(0, dtype="float64")
        return frame.values(field_name, n)

    def prediction_for(self, symbol: str) -> float | None:
        if self.predictions is None:
            return None
        return self.predictions.lookup(symbol, self.ts, self.as_of)


class Strategy(ABC):
    """Base class for every strategy."""

    name: str = "strategy"

    @abstractmethod
    def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:
        """Target weights for the next execution.  Missing symbols mean 'go flat'."""

    def warmup_bars(self) -> int:
        """How many bars of history the strategy needs before it should trade."""
        return 0

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": type(self).__name__}


# ---------------------------------------------------------------------------
# Reference strategies
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BuyAndHold(Strategy):
    """Constant target weights.  The benchmark every other strategy must beat.

    Note that it still rebalances: holding a constant *weight* requires trading
    as prices move.  ``rebalance_tolerance`` suppresses trades below a drift
    threshold so the benchmark is not penalised by daily churn it would not
    incur in practice.
    """

    symbols: Sequence[str]
    weights: Mapping[str, float] | None = None
    name: str = "buy_and_hold"

    def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:  # noqa: ARG002
        if self.weights:
            return [TargetWeight(s, float(w), "fixed") for s, w in self.weights.items()]
        share = 1.0 / len(self.symbols)
        return [TargetWeight(s, share, "equal_weight") for s in self.symbols]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "BuyAndHold",
            "symbols": list(self.symbols),
            "weights": dict(self.weights) if self.weights else None,
        }


@dataclass(slots=True)
class MovingAverageCrossover(Strategy):
    """Long when the fast moving average is above the slow one, else flat or short.

    The classic textbook strategy, included because it is the right control: if
    a machine-learning model cannot beat this after costs, that is the finding.
    """

    symbols: Sequence[str]
    fast: int = 20
    slow: int = 100
    allow_short: bool = False
    price_field: str = "adj_close"
    name: str = "ma_crossover"

    def warmup_bars(self) -> int:
        return self.slow + 1

    def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:
        targets: list[TargetWeight] = []
        live = [s for s in self.symbols if s in ctx.prices]
        if not live:
            return targets
        share = 1.0 / len(live)
        for symbol in live:
            closes = ctx.recent(symbol, self.slow, self.price_field)
            if len(closes) < self.slow:
                continue
            fast_ma = float(closes[-self.fast :].mean())
            slow_ma = float(closes.mean())
            if fast_ma > slow_ma:
                targets.append(TargetWeight(symbol, share, "fast_above_slow"))
            elif self.allow_short:
                targets.append(TargetWeight(symbol, -share, "fast_below_slow"))
            else:
                targets.append(TargetWeight(symbol, 0.0, "fast_below_slow"))
        return targets

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "MovingAverageCrossover",
            "symbols": list(self.symbols),
            "fast": self.fast,
            "slow": self.slow,
            "allow_short": self.allow_short,
        }


@dataclass(slots=True)
class TimeSeriesMomentum(Strategy):
    """Long if the trailing ``lookback``-day return is positive.

    Included because time-series momentum is one of the few effects with
    out-of-sample evidence across decades and asset classes, so it is a fair
    statistical baseline rather than a straw man.
    """

    symbols: Sequence[str]
    lookback: int = 126
    allow_short: bool = True
    price_field: str = "adj_close"
    name: str = "ts_momentum"

    def warmup_bars(self) -> int:
        return self.lookback + 1

    def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:
        targets: list[TargetWeight] = []
        live = [s for s in self.symbols if s in ctx.prices]
        if not live:
            return targets
        share = 1.0 / len(live)
        for symbol in live:
            closes = ctx.recent(symbol, self.lookback + 1, self.price_field)
            if len(closes) < self.lookback + 1:
                continue
            trailing = float(closes[-1] / closes[0] - 1.0)
            if trailing > 0:
                targets.append(TargetWeight(symbol, share, "positive_momentum"))
            elif self.allow_short:
                targets.append(TargetWeight(symbol, -share, "negative_momentum"))
            else:
                targets.append(TargetWeight(symbol, 0.0, "negative_momentum"))
        return targets

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "TimeSeriesMomentum",
            "symbols": list(self.symbols),
            "lookback": self.lookback,
            "allow_short": self.allow_short,
        }


@dataclass(slots=True)
class PredictionStrategy(Strategy):
    """Turn model forecasts into positions.

    The predictions frame is supplied up front and truncated to the decision
    instant by the engine, so a model whose forecasts were produced with future
    data will still be blocked by the availability filter -- but only if the
    prediction rows carry an honest ``available_at``.  The walk-forward runner
    sets that to the decision bar's availability, so an in-sample forecast can
    never be consumed as if it had been out-of-sample.

    ``long_threshold`` and ``short_threshold`` are on the predicted quantity
    (typically a forward return), not on a rank, so the strategy is explicit
    about how much predicted edge is needed to justify a trade.
    """

    symbols: Sequence[str]
    long_threshold: float = 0.0
    short_threshold: float | None = None
    max_positions: int | None = None
    gross_leverage: float = 1.0
    name: str = "prediction"

    def on_bar(self, ctx: DecisionContext) -> Sequence[TargetWeight]:
        scores: dict[str, float] = {}
        for symbol in self.symbols:
            value = ctx.prediction_for(symbol)
            if value is not None and np.isfinite(value):
                scores[symbol] = value
        if not scores:
            return []

        longs = {s: v for s, v in scores.items() if v > self.long_threshold}
        shorts: dict[str, float] = {}
        if self.short_threshold is not None:
            shorts = {s: v for s, v in scores.items() if v < self.short_threshold}

        if self.max_positions is not None:
            longs = dict(sorted(longs.items(), key=lambda kv: -kv[1])[: self.max_positions])
            shorts = dict(sorted(shorts.items(), key=lambda kv: kv[1])[: self.max_positions])

        n = len(longs) + len(shorts)
        if n == 0:
            return [TargetWeight(s, 0.0, "no_signal") for s in self.symbols]
        share = self.gross_leverage / n
        targets = [TargetWeight(s, share, "predicted_up") for s in longs]
        targets += [TargetWeight(s, -share, "predicted_down") for s in shorts]
        flat = set(self.symbols) - set(longs) - set(shorts)
        targets += [TargetWeight(s, 0.0, "below_threshold") for s in flat]
        return targets

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "PredictionStrategy",
            "symbols": list(self.symbols),
            "long_threshold": self.long_threshold,
            "short_threshold": self.short_threshold,
            "max_positions": self.max_positions,
            "gross_leverage": self.gross_leverage,
        }


__all__ = [
    "BuyAndHold",
    "DecisionContext",
    "MovingAverageCrossover",
    "PortfolioView",
    "PredictionBook",
    "PredictionStrategy",
    "Strategy",
    "TimeSeriesMomentum",
]

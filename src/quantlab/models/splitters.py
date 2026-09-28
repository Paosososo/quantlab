"""Time-series cross-validation.

Why not ``train_test_split`` or ``KFold``
-----------------------------------------
Random splitting assumes observations are exchangeable.  Financial time series
are not: they are autocorrelated, non-stationary, and the thing being predicted
is the future.  A random split trains on 2023 and tests on 2019, which is not a
forecast, it is interpolation.  Published results using random splits on
financial data are, without exception, measuring the wrong thing.  So this
module offers no shuffling option at all -- not as a default, not as a flag.

Three mechanisms, each fixing a different problem
-------------------------------------------------
**Chronological folds.**  Training always precedes testing.  Expanding windows
grow the training set each fold (uses all history, assumes the relationship is
stable); rolling windows keep it fixed (adapts to regime change, uses less
data).  Both are provided because which is right is an empirical question about
the data, not a settled one.

**Purging.**  With an ``h``-day forward return, the label for a training day
near the fold boundary is realised *inside* the test window.  Training on it
means the model has seen test-period outcomes.  Purging drops every training row
whose label became knowable at or after the test window opens.  This is why
:func:`quantlab.features.pipeline.build_labels` carries
``label_available_at``: without it, purging has to guess the horizon.

**Embargo.**  Even after purging, serial correlation means the observations
immediately before the test window are nearly the same information as the
observations inside it.  Dropping a further gap reduces that contamination.
Both ideas are from Lopez de Prado, *Advances in Financial Machine Learning*
(2018), chapter 7.

Splits are defined over unique **timestamps**, not row positions, so a panel of
several symbols cuts cleanly: every symbol's rows for a given date land on the
same side of every boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from quantlab.exceptions import InsufficientDataError, LookAheadError

TIME_COLUMN = "ts"
LABEL_AVAILABILITY_COLUMN = "label_available_at"


@dataclass(frozen=True, slots=True)
class Split:
    """One fold: positional indices into the frame that was split."""

    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_purged: int = 0
    n_embargoed: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "n_train": int(len(self.train_index)),
            "n_test": int(len(self.test_index)),
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
            "n_purged": self.n_purged,
            "n_embargoed": self.n_embargoed,
        }


@dataclass(frozen=True, slots=True)
class SplitterConfig:
    n_splits: int = 5
    #: Number of distinct timestamps in each test window.
    test_size: int = 126
    #: Minimum distinct timestamps required before the first fold trains.
    min_train_size: int = 504
    #: Extra gap between the end of training and the start of testing.
    embargo: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(days=0))
    purge: bool = True


class TimeSeriesSplitter(ABC):
    """Base class.  ``split`` consumes the ``meta`` frame from the feature pipeline."""

    name: str = "abstract"

    def __init__(self, config: SplitterConfig | None = None, **overrides: Any) -> None:
        base = config or SplitterConfig()
        if overrides:
            # ``replace`` rather than ``**config.__dict__``: SplitterConfig uses
            # slots, so it has no ``__dict__`` to unpack.
            base = replace(base, **overrides)
        self.config = base

    @abstractmethod
    def _boundaries(self, n_timestamps: int) -> list[tuple[int, int, int, int]]:
        """(train_lo, train_hi, test_lo, test_hi) as half-open timestamp positions."""

    def split(self, meta: pd.DataFrame) -> Iterator[Split]:
        if TIME_COLUMN not in meta.columns:
            raise ValueError(f"meta frame must contain a {TIME_COLUMN!r} column")

        times = pd.to_datetime(meta[TIME_COLUMN], utc=True)
        unique_times = pd.DatetimeIndex(np.sort(times.unique()))
        boundaries = self._boundaries(len(unique_times))
        if not boundaries:
            raise InsufficientDataError(
                "not enough distinct timestamps for the requested split",
                have=len(unique_times),
                need=self.config.min_train_size + self.config.test_size,
                splitter=self.name,
            )

        label_available = (
            pd.to_datetime(meta[LABEL_AVAILABILITY_COLUMN], utc=True)
            if LABEL_AVAILABILITY_COLUMN in meta.columns
            else None
        )

        for fold, (train_lo, train_hi, test_lo, test_hi) in enumerate(boundaries):
            train_from = unique_times[train_lo]
            train_to = unique_times[train_hi - 1]
            test_from = unique_times[test_lo]
            test_to = unique_times[test_hi - 1]

            train_mask = (times >= train_from) & (times <= train_to)
            test_mask = (times >= test_from) & (times <= test_to)

            n_embargoed = 0
            if self.config.embargo > pd.Timedelta(0):
                cutoff = test_from - self.config.embargo
                embargoed = train_mask & (times > cutoff)
                n_embargoed = int(embargoed.sum())
                train_mask = train_mask & ~embargoed

            n_purged = 0
            if self.config.purge and label_available is not None:
                # Drop training rows whose label is only knowable once the test
                # window has already begun.
                leaking = train_mask & (label_available >= test_from)
                n_purged = int(leaking.sum())
                train_mask = train_mask & ~leaking

            train_index = np.flatnonzero(train_mask.to_numpy())
            test_index = np.flatnonzero(test_mask.to_numpy())
            if len(train_index) == 0 or len(test_index) == 0:
                continue

            actual_train_end = times.iloc[train_index].max()
            if actual_train_end >= test_from:
                raise LookAheadError(
                    "training data reaches into the test window after purging",
                    fold=fold,
                    train_end=str(actual_train_end),
                    test_start=str(test_from),
                )

            yield Split(
                fold=fold,
                train_index=train_index,
                test_index=test_index,
                train_start=train_from,
                train_end=actual_train_end,
                test_start=test_from,
                test_end=test_to,
                n_purged=n_purged,
                n_embargoed=n_embargoed,
            )

    def describe(self) -> dict[str, Any]:
        return {
            "splitter": self.name,
            "n_splits": self.config.n_splits,
            "test_size": self.config.test_size,
            "min_train_size": self.config.min_train_size,
            "embargo_days": self.config.embargo.days,
            "purge": self.config.purge,
        }


class ExpandingWindowSplitter(TimeSeriesSplitter):
    """Training set grows with each fold; test windows are consecutive.

    Fold k trains on everything up to the start of test window k.  This is the
    right default when you believe the relationship being modelled is stable,
    because it uses every observation available at each decision point -- which
    is also what a real research process would have had.
    """

    name = "expanding_window"

    def _boundaries(self, n_timestamps: int) -> list[tuple[int, int, int, int]]:
        cfg = self.config
        needed = cfg.min_train_size + cfg.test_size
        if n_timestamps < needed:
            return []
        available = n_timestamps - cfg.min_train_size
        n_folds = min(cfg.n_splits, available // cfg.test_size)
        if n_folds <= 0:
            return []
        # Anchor the last fold at the end of the sample so the most recent data
        # is always tested; earlier folds step backwards from there.
        out: list[tuple[int, int, int, int]] = []
        for k in range(n_folds):
            test_hi = n_timestamps - (n_folds - 1 - k) * cfg.test_size
            test_lo = test_hi - cfg.test_size
            out.append((0, test_lo, test_lo, test_hi))
        return out


class RollingWindowSplitter(TimeSeriesSplitter):
    """Fixed-length training window that slides forward.

    Use when the data-generating process is suspected to change: a 2008 regime
    is arguably worse than useless for predicting 2021, and an expanding window
    cannot forget it.  The cost is fewer training observations per fold.
    """

    name = "rolling_window"

    def __init__(
        self, config: SplitterConfig | None = None, train_size: int = 504, **overrides: Any
    ) -> None:
        super().__init__(config, **overrides)
        self.train_size = train_size

    def _boundaries(self, n_timestamps: int) -> list[tuple[int, int, int, int]]:
        cfg = self.config
        window = max(self.train_size, cfg.min_train_size)
        if n_timestamps < window + cfg.test_size:
            return []
        n_folds = min(cfg.n_splits, (n_timestamps - window) // cfg.test_size)
        if n_folds <= 0:
            return []
        out: list[tuple[int, int, int, int]] = []
        for k in range(n_folds):
            test_hi = n_timestamps - (n_folds - 1 - k) * cfg.test_size
            test_lo = test_hi - cfg.test_size
            train_lo = max(0, test_lo - window)
            out.append((train_lo, test_lo, test_lo, test_hi))
        return out

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "train_size": self.train_size}


class SingleHoldoutSplitter(TimeSeriesSplitter):
    """One chronological train/test cut.

    Provided for the final, once-only evaluation.  Repeatedly tuning against a
    walk-forward score is itself a form of overfitting; a holdout that is looked
    at once is the honest final number.
    """

    name = "single_holdout"

    def __init__(
        self, config: SplitterConfig | None = None, test_fraction: float = 0.2, **overrides: Any
    ) -> None:
        super().__init__(config, **overrides)
        self.test_fraction = test_fraction

    def _boundaries(self, n_timestamps: int) -> list[tuple[int, int, int, int]]:
        test_size = max(1, int(n_timestamps * self.test_fraction))
        train_size = n_timestamps - test_size
        if train_size < max(1, self.config.min_train_size // 4):
            return []
        return [(0, train_size, train_size, n_timestamps)]

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "test_fraction": self.test_fraction}


def get_splitter(name: str, **kwargs: Any) -> TimeSeriesSplitter:
    registry = {
        "expanding_window": ExpandingWindowSplitter,
        "rolling_window": RollingWindowSplitter,
        "single_holdout": SingleHoldoutSplitter,
    }
    try:
        cls = registry[name]
    except KeyError as exc:
        raise ValueError(f"unknown splitter {name!r}; known: {sorted(registry)}") from exc
    return cls(**kwargs)


__all__ = [
    "ExpandingWindowSplitter",
    "RollingWindowSplitter",
    "SingleHoldoutSplitter",
    "Split",
    "SplitterConfig",
    "TimeSeriesSplitter",
    "get_splitter",
]

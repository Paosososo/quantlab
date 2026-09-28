"""Time-series splitters: no fold may leak the future into training."""

from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd
import pytest

from quantlab.exceptions import InsufficientDataError
from quantlab.models.splitters import (
    ExpandingWindowSplitter,
    RollingWindowSplitter,
    SingleHoldoutSplitter,
    SplitterConfig,
    get_splitter,
)

pytestmark = [pytest.mark.leakage, pytest.mark.unit]


def make_meta(
    n: int = 1500, symbols: tuple[str, ...] = ("A",), horizon_days: int = 7
) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    frames = [
        pd.DataFrame(
            {
                "symbol": symbol,
                "ts": dates,
                "available_at": dates + pd.Timedelta(hours=21),
                "label_available_at": dates + pd.Timedelta(days=horizon_days),
            }
        )
        for symbol in symbols
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)


CONFIG = SplitterConfig(n_splits=4, test_size=126, min_train_size=504)


class TestOrdering:
    @pytest.mark.parametrize(
        "splitter",
        [
            ExpandingWindowSplitter(CONFIG),
            RollingWindowSplitter(CONFIG, train_size=504),
            SingleHoldoutSplitter(CONFIG, test_fraction=0.25),
        ],
    )
    def test_training_always_precedes_testing(self, splitter):
        meta = make_meta()
        times = pd.to_datetime(meta["ts"], utc=True)
        folds = list(splitter.split(meta))
        assert folds
        for split in folds:
            assert times.iloc[split.train_index].max() < times.iloc[split.test_index].min()

    def test_train_and_test_never_share_a_row(self):
        meta = make_meta()
        for split in ExpandingWindowSplitter(CONFIG).split(meta):
            assert not set(split.train_index) & set(split.test_index)

    def test_folds_move_forward_in_time(self):
        meta = make_meta()
        folds = list(ExpandingWindowSplitter(CONFIG).split(meta))
        starts = [f.test_start for f in folds]
        assert starts == sorted(starts)

    def test_test_windows_do_not_overlap(self):
        meta = make_meta()
        folds = list(ExpandingWindowSplitter(CONFIG).split(meta))
        for earlier, later in itertools.pairwise(folds):
            assert earlier.test_end < later.test_start


class TestPurging:
    def test_overlapping_labels_are_purged(self):
        """A 30-day label near the boundary resolves inside the test window."""
        meta = make_meta(horizon_days=30)
        folds = list(ExpandingWindowSplitter(CONFIG).split(meta))
        assert all(f.n_purged > 0 for f in folds)

    def test_purging_removes_exactly_the_leaking_rows(self):
        meta = make_meta(horizon_days=30)
        label_available = pd.to_datetime(meta["label_available_at"], utc=True)
        for split in ExpandingWindowSplitter(CONFIG).split(meta):
            kept = label_available.iloc[split.train_index]
            assert (kept < split.test_start).all()

    def test_a_one_day_label_purges_little(self):
        meta = make_meta(horizon_days=1)
        folds = list(ExpandingWindowSplitter(CONFIG).split(meta))
        assert all(f.n_purged <= 2 for f in folds)

    def test_purging_can_be_disabled_for_comparison(self):
        """Kept as an option only so the effect of purging can be measured."""
        meta = make_meta(horizon_days=30)
        with_purge = list(ExpandingWindowSplitter(replace(CONFIG, purge=True)).split(meta))
        without = list(ExpandingWindowSplitter(replace(CONFIG, purge=False)).split(meta))
        assert len(with_purge[0].train_index) < len(without[0].train_index)


class TestEmbargo:
    def test_the_embargo_removes_the_rows_before_the_test_window(self):
        config = SplitterConfig(
            n_splits=3, test_size=126, min_train_size=504, embargo=pd.Timedelta(days=10)
        )
        meta = make_meta()
        times = pd.to_datetime(meta["ts"], utc=True)
        for split in ExpandingWindowSplitter(config).split(meta):
            gap = split.test_start - times.iloc[split.train_index].max()
            # The embargo is inclusive: the last training row is at least
            # `embargo` before the test window opens.
            assert gap >= pd.Timedelta(days=10)
            assert split.n_embargoed > 0

    def test_no_embargo_leaves_the_boundary_tight(self):
        meta = make_meta(horizon_days=1)
        for split in ExpandingWindowSplitter(CONFIG).split(meta):
            assert split.n_embargoed == 0


class TestWindowShapes:
    def test_expanding_windows_grow(self):
        meta = make_meta()
        sizes = [len(s.train_index) for s in ExpandingWindowSplitter(CONFIG).split(meta)]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_rolling_windows_stay_the_same_size(self):
        meta = make_meta()
        sizes = [
            len(s.train_index) for s in RollingWindowSplitter(CONFIG, train_size=504).split(meta)
        ]
        assert max(sizes) - min(sizes) <= 5  # only purging/embargo may vary it

    def test_panel_data_splits_on_dates_not_rows(self):
        """Every symbol's rows for a date must land on the same side of a cut."""
        meta = make_meta(symbols=("A", "B", "C"))
        times = pd.to_datetime(meta["ts"], utc=True)
        for split in ExpandingWindowSplitter(CONFIG).split(meta):
            train_dates = set(times.iloc[split.train_index])
            test_dates = set(times.iloc[split.test_index])
            assert not train_dates & test_dates
            # Each test date should contribute all three symbols.
            counts = times.iloc[split.test_index].value_counts()
            assert set(counts.unique()) == {3}

    def test_single_holdout_yields_one_fold(self):
        meta = make_meta()
        folds = list(SingleHoldoutSplitter(CONFIG, test_fraction=0.2).split(meta))
        assert len(folds) == 1
        assert len(folds[0].test_index) == pytest.approx(len(meta) * 0.2, rel=0.02)


class TestGuards:
    def test_too_little_data_raises(self):
        meta = make_meta(n=100)
        with pytest.raises(InsufficientDataError):
            list(ExpandingWindowSplitter(CONFIG).split(meta))

    def test_a_missing_time_column_raises(self):
        with pytest.raises(ValueError, match="ts"):
            list(ExpandingWindowSplitter(CONFIG).split(pd.DataFrame({"x": [1, 2, 3]})))

    def test_the_factory_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="unknown splitter"):
            get_splitter("shuffle_split")

    def test_there_is_no_shuffle_option_anywhere(self):
        """Random splitting is not offered, by design.

        This test is a guard against a future contributor adding one 'for
        convenience'.  On financial time series it is not a convenience, it is a
        wrong answer.
        """
        for splitter in (ExpandingWindowSplitter(CONFIG), RollingWindowSplitter(CONFIG)):
            assert "shuffle" not in splitter.describe()
            assert not hasattr(splitter, "shuffle")

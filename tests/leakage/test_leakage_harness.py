"""Tests for the leakage detector itself.

A leak detector that never fires is worthless, and one that always fires gets
switched off.  So this file has three parts: known-leaky functions that must be
caught, known-causal functions that must pass, and the production feature
library which must pass.

There is also a behavioural test for rolling skew, which must pass regardless
of pandas' internal numerical implementation.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import LookAheadError
from quantlab.features.guards import (
    assert_availability_ordering,
    assert_causal,
    assert_split_ordering,
    corrupt_future,
    punch_gap,
)
from quantlab.features.library import default_feature_set
from quantlab.features.pipeline import compute_symbol_features

pytestmark = [pytest.mark.leakage, pytest.mark.unit]

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


# ---------------------------------------------------------------------------
# Deliberately broken feature functions
# ---------------------------------------------------------------------------
def leak_centered_rolling(frame: pd.DataFrame) -> pd.DataFrame:
    """Half the window is in the future."""
    return pd.DataFrame({"x": frame["adj_close"].rolling(21, min_periods=21, center=True).mean()})


def leak_whole_sample_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation computed over the entire history."""
    close = frame["adj_close"]
    return pd.DataFrame({"x": (close - close.mean()) / close.std()})


def leak_backfill(frame: pd.DataFrame) -> pd.DataFrame:
    """Gaps filled from later observations."""
    return pd.DataFrame({"x": frame["adj_close"].bfill()})


def leak_interpolate_both(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"x": frame["adj_close"].interpolate(limit_direction="both")})


def leak_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Tomorrow's price as a feature: the purest form of target leakage."""
    return pd.DataFrame({"x": frame["adj_close"].shift(-1)})


def leak_scale_by_global_max(frame: pd.DataFrame) -> pd.DataFrame:
    """Min-max scaling fitted on the whole sample, the classic preprocessing leak."""
    close = frame["adj_close"]
    return pd.DataFrame({"x": close / close.max()})


def leak_reverse_rolling(frame: pd.DataFrame) -> pd.DataFrame:
    """A rolling window applied to the reversed series."""
    reversed_close = frame["adj_close"][::-1]
    return pd.DataFrame({"x": reversed_close.rolling(10, min_periods=10).min()[::-1]})


def leak_expanding_from_the_end(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["adj_close"]
    return pd.DataFrame({"x": close[::-1].expanding().mean()[::-1]})


# ---------------------------------------------------------------------------
# Correct feature functions
# ---------------------------------------------------------------------------
def causal_rolling_mean(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"x": frame["adj_close"].rolling(21, min_periods=21).mean()})


def causal_ewm(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"x": frame["adj_close"].ewm(span=20, adjust=False, min_periods=20).mean()})


def causal_forward_fill(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"x": frame["adj_close"].ffill()})


def causal_lagged_return(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["adj_close"]
    return pd.DataFrame({"x": close.pct_change(5, fill_method=None).shift(1)})


LEAKY = {
    "centered_rolling": leak_centered_rolling,
    "whole_sample_zscore": leak_whole_sample_zscore,
    "backfill": leak_backfill,
    "interpolate_both": leak_interpolate_both,
    "target_shift": leak_target,
    "scale_by_global_max": leak_scale_by_global_max,
    "reverse_rolling": leak_reverse_rolling,
    "expanding_from_end": leak_expanding_from_the_end,
}

CAUSAL = {
    "rolling_mean": causal_rolling_mean,
    "ewm": causal_ewm,
    "forward_fill": causal_forward_fill,
    "lagged_return": causal_lagged_return,
}


class TestHarnessCatchesLeaks:
    @pytest.mark.parametrize("name", sorted(LEAKY))
    def test_known_leak_is_detected(self, name, price_frame):
        report = assert_causal(
            LEAKY[name], price_frame, input_columns=["adj_close"], raise_on_finding=False
        )
        assert not report.clean, f"{name} reads the future but was not detected"
        assert report.findings[0].describe()

    @pytest.mark.parametrize("name", sorted(LEAKY))
    def test_known_leak_raises_by_default(self, name, price_frame):
        with pytest.raises(LookAheadError):
            assert_causal(LEAKY[name], price_frame, input_columns=["adj_close"])


class TestHarnessAcceptsCausalCode:
    @pytest.mark.parametrize("name", sorted(CAUSAL))
    def test_causal_function_passes(self, name, price_frame):
        report = assert_causal(CAUSAL[name], price_frame, input_columns=["adj_close"])
        assert report.clean

    def test_production_feature_library_is_clean(self, price_frame):
        """The whole point.  If this ever fails, a feature started reading ahead."""
        feature_set = default_feature_set(include_macro=False)
        compute = functools.partial(compute_symbol_features, feature_set=feature_set)
        report = assert_causal(compute, price_frame, input_columns=PRICE_COLUMNS)
        assert report.clean
        assert set(report.checked_features) == set(feature_set.names)
        assert len(report.cut_points) >= 2


class TestCorruptionModel:
    def test_corruption_leaves_the_head_untouched(self, price_frame):
        cut = len(price_frame) // 2
        corrupted = corrupt_future(price_frame, cut, numeric_columns=PRICE_COLUMNS)
        pd.testing.assert_frame_equal(
            price_frame.iloc[: cut + 1][PRICE_COLUMNS],
            corrupted.iloc[: cut + 1][PRICE_COLUMNS],
        )

    def test_corruption_actually_changes_the_tail(self, price_frame):
        cut = len(price_frame) // 2
        corrupted = corrupt_future(price_frame, cut, numeric_columns=PRICE_COLUMNS)
        tail_before = price_frame["adj_close"].iloc[cut + 1 :].to_numpy()
        tail_after = corrupted["adj_close"].iloc[cut + 1 :].to_numpy()
        assert not np.allclose(tail_before, tail_after, equal_nan=True)

    def test_corruption_injects_nans(self, price_frame):
        cut = len(price_frame) // 2
        corrupted = corrupt_future(
            price_frame, cut, numeric_columns=PRICE_COLUMNS, nan_fraction=0.3
        )
        assert corrupted["adj_close"].iloc[cut + 1 :].isna().any()

    def test_punch_gap_blanks_the_rows_before_the_cut(self, price_frame):
        cut = 100
        gapped = punch_gap(price_frame, cut, numeric_columns=PRICE_COLUMNS, width=3)
        assert gapped["adj_close"].iloc[98:101].isna().all()
        assert gapped["adj_close"].iloc[:98].notna().all()

    def test_causal_rolling_skew_passes_across_pandas_versions(self):
        """Check the leakage contract, not a pandas implementation detail."""
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({"adj_close": rng.normal(0, 0.01, 800)})

        def rolling_skew(values: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"skew": values["adj_close"].rolling(63, min_periods=63).skew()})

        report = assert_causal(rolling_skew, frame, input_columns=["adj_close"])
        assert report.clean


class TestAvailabilityAssertions:
    def test_availability_before_observation_is_rejected(self):
        index = pd.date_range("2020-01-01", periods=5, tz="UTC")
        frame = pd.DataFrame({"ts": index, "available_at": index - pd.Timedelta(days=1)})
        with pytest.raises(LookAheadError, match="available before"):
            assert_availability_ordering(frame)

    def test_valid_availability_passes(self, price_frame):
        assert_availability_ordering(price_frame)

    def test_missing_columns_raise(self):
        with pytest.raises(LookAheadError):
            assert_availability_ordering(pd.DataFrame({"x": [1, 2]}))


class TestSplitOrdering:
    def test_overlapping_train_and_test_is_rejected(self):
        train = pd.date_range("2020-01-01", periods=100, tz="UTC")
        test = pd.date_range("2020-03-01", periods=50, tz="UTC")
        with pytest.raises(LookAheadError):
            assert_split_ordering(train, test)

    def test_a_gap_smaller_than_the_embargo_is_rejected(self):
        train = pd.date_range("2020-01-01", periods=60, tz="UTC")
        test = pd.date_range("2020-03-02", periods=30, tz="UTC")
        with pytest.raises(LookAheadError):
            assert_split_ordering(train, test, embargo=pd.Timedelta(days=30))

    def test_a_sufficient_embargo_passes(self):
        train = pd.date_range("2020-01-01", periods=60, tz="UTC")
        test = pd.date_range("2020-06-01", periods=30, tz="UTC")
        assert_split_ordering(train, test, embargo=pd.Timedelta(days=10))

    def test_empty_splits_are_ignored(self):
        empty = pd.DatetimeIndex([], tz="UTC")
        assert_split_ordering(empty, pd.date_range("2020-01-01", periods=3, tz="UTC"))

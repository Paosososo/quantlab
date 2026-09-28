"""Timestamp normalisation and the availability model."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from quantlab.exceptions import NonMonotonicTimestampError, TemporalError
from quantlab.timeutils import (
    US_EQUITY_SESSION,
    UTC,
    as_of_filter,
    assert_monotonic,
    ensure_utc_index,
    get_session,
    infer_periods_per_year,
    macro_available_at,
    normalise_index,
    price_available_at,
    session_date_to_ts,
    to_utc,
)

pytestmark = pytest.mark.unit


class TestToUtc:
    def test_naive_datetime_is_rejected(self):
        """Guessing the zone of a naive timestamp is how series get misaligned."""
        with pytest.raises(TemporalError, match="naive datetime"):
            to_utc(dt.datetime(2020, 1, 1, 12, 0))

    def test_aware_datetime_is_converted(self):
        bangkok = dt.timezone(dt.timedelta(hours=7))
        value = dt.datetime(2020, 1, 1, 7, 0, tzinfo=bangkok)
        assert to_utc(value) == dt.datetime(2020, 1, 1, 0, 0, tzinfo=UTC)

    def test_date_becomes_utc_midnight(self):
        assert to_utc(dt.date(2020, 6, 1)) == dt.datetime(2020, 6, 1, tzinfo=UTC)

    def test_string_and_timestamp_accepted(self):
        assert to_utc("2020-06-01T00:00:00+00:00") == dt.datetime(2020, 6, 1, tzinfo=UTC)
        assert to_utc(pd.Timestamp("2020-06-01", tz="UTC")) == dt.datetime(2020, 6, 1, tzinfo=UTC)

    def test_unsupported_type_raises(self):
        with pytest.raises(TemporalError):
            to_utc(12345)  # type: ignore[arg-type]


class TestAvailability:
    def test_price_availability_is_the_session_close_in_utc(self):
        """A January bar for a New York session is knowable at 21:00 UTC (16:00 EST)."""
        available = price_available_at(dt.date(2020, 1, 6), US_EQUITY_SESSION)
        assert available == dt.datetime(2020, 1, 6, 21, 0, tzinfo=UTC)

    def test_price_availability_follows_daylight_saving(self):
        """In July, 16:00 EDT is 20:00 UTC, not 21:00.  A fixed offset would be wrong."""
        available = price_available_at(dt.date(2020, 7, 6), US_EQUITY_SESSION)
        assert available == dt.datetime(2020, 7, 6, 20, 0, tzinfo=UTC)

    def test_availability_is_never_before_the_observation(self):
        session_date = dt.date(2020, 3, 10)
        assert price_available_at(session_date) >= session_date_to_ts(session_date)

    def test_macro_availability_applies_the_publication_lag(self):
        available = macro_available_at(dt.date(2020, 1, 1), 45)
        assert available.date() == dt.date(2020, 2, 15)

    def test_negative_macro_lag_is_rejected(self):
        with pytest.raises(TemporalError):
            macro_available_at(dt.date(2020, 1, 1), -1)

    def test_as_of_filter_hides_unpublished_rows(self):
        index = pd.date_range("2020-01-01", periods=5, tz="UTC")
        frame = pd.DataFrame(
            {"ts": index, "available_at": index + pd.Timedelta(hours=21), "value": range(5)}
        )
        visible = as_of_filter(frame, dt.datetime(2020, 1, 3, 21, tzinfo=UTC))
        assert len(visible) == 3
        assert visible["value"].tolist() == [0, 1, 2]

    def test_as_of_filter_requires_the_availability_column(self):
        frame = pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=2, tz="UTC")})
        with pytest.raises(TemporalError):
            as_of_filter(frame, dt.datetime(2020, 1, 3, tzinfo=UTC))


class TestNormalisation:
    def test_normalise_index_sorts_and_localises(self):
        frame = pd.DataFrame({"ts": ["2020-01-03", "2020-01-01", "2020-01-02"], "v": [3, 1, 2]})
        out = normalise_index(frame, "ts")
        assert out["v"].tolist() == [1, 2, 3]
        assert str(out["ts"].dt.tz) == "UTC"

    def test_normalise_index_rejects_duplicates(self):
        frame = pd.DataFrame({"ts": ["2020-01-01", "2020-01-01"], "v": [1, 2]})
        with pytest.raises(NonMonotonicTimestampError):
            normalise_index(frame, "ts")

    def test_assert_monotonic_catches_unsorted(self):
        with pytest.raises(NonMonotonicTimestampError):
            assert_monotonic(pd.DatetimeIndex(["2020-01-02", "2020-01-01"]))

    def test_ensure_utc_index_rejects_naive_index(self):
        frame = pd.DataFrame({"v": [1, 2]}, index=pd.date_range("2020-01-01", periods=2))
        with pytest.raises(TemporalError):
            ensure_utc_index(frame)

    def test_ensure_utc_index_converts_other_zones(self):
        index = pd.date_range("2020-01-01", periods=2, tz="Asia/Bangkok")
        out = ensure_utc_index(pd.DataFrame({"v": [1, 2]}, index=index))
        assert str(out.index.tz) == "UTC"


class TestCalendarHelpers:
    @pytest.mark.parametrize(
        ("freq", "expected"),
        [("B", 252), ("W", 52), ("MS", 12), ("QS", 4)],
    )
    def test_infer_periods_per_year(self, freq, expected):
        index = pd.date_range("2015-01-01", periods=40, freq=freq, tz="UTC")
        assert infer_periods_per_year(index) == expected

    def test_infer_falls_back_on_short_series(self):
        index = pd.DatetimeIndex(["2020-01-01"], tz="UTC")
        assert infer_periods_per_year(index, default=99) == 99

    def test_unknown_session_raises(self):
        with pytest.raises(TemporalError):
            get_session("mars_exchange")

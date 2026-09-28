"""Point-in-time views must make the future unreachable, not merely discouraged."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from quantlab.exceptions import LookAheadError
from quantlab.pit import PointInTimeFrame, PointInTimeUniverse
from quantlab.timeutils import UTC

pytestmark = pytest.mark.unit


@pytest.fixture
def frame() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=10, tz="UTC")
    return pd.DataFrame(
        {
            "ts": index,
            "available_at": index + pd.Timedelta(hours=21),
            "close": [float(i) for i in range(10)],
        }
    )


class TestPointInTimeFrame:
    def test_only_published_rows_are_visible(self, frame):
        view = PointInTimeFrame(frame, dt.datetime(2020, 1, 5, 21, tzinfo=UTC))
        assert len(view) == 5
        assert view.latest("close") == 4.0

    def test_a_row_published_one_second_later_is_invisible(self, frame):
        """The boundary is inclusive on availability and nothing else."""
        view = PointInTimeFrame(frame, dt.datetime(2020, 1, 5, 20, 59, 59, tzinfo=UTC))
        assert len(view) == 4

    def test_mutating_the_view_cannot_affect_the_source(self, frame):
        view = PointInTimeFrame(frame, dt.datetime(2020, 1, 5, 21, tzinfo=UTC))
        copy = view.frame()
        copy.loc[0, "close"] = 999.0
        assert frame.loc[0, "close"] == 0.0
        assert view.latest("close") == 4.0

    def test_reading_a_future_timestamp_raises(self, frame):
        view = PointInTimeFrame(frame, dt.datetime(2020, 1, 5, 21, tzinfo=UTC))
        with pytest.raises(LookAheadError):
            view.value_at(dt.datetime(2020, 1, 9, tzinfo=UTC), "close")

    def test_missing_availability_column_is_refused(self):
        frame = pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=3, tz="UTC")})
        with pytest.raises(LookAheadError, match="availability column"):
            PointInTimeFrame(frame, dt.datetime(2020, 1, 2, tzinfo=UTC))

    def test_require_history_raises_when_warmup_is_short(self, frame):
        view = PointInTimeFrame(frame, dt.datetime(2020, 1, 2, 21, tzinfo=UTC))
        with pytest.raises(LookAheadError, match="insufficient visible history"):
            view.require_history(50)

    def test_tail_returns_the_most_recent_rows(self, frame):
        view = PointInTimeFrame(frame, dt.datetime(2020, 1, 8, 21, tzinfo=UTC))
        assert view.tail(3)["close"].tolist() == [5.0, 6.0, 7.0]

    def test_empty_view_before_any_publication(self, frame):
        view = PointInTimeFrame(frame, dt.datetime(2019, 1, 1, tzinfo=UTC))
        assert len(view) == 0
        assert view.latest() is None


class TestPointInTimeUniverse:
    def test_groups_by_symbol_and_shares_the_as_of(self, frame):
        tidy = pd.concat([frame.assign(symbol="A"), frame.assign(symbol="B")], ignore_index=True)
        as_of = dt.datetime(2020, 1, 5, 21, tzinfo=UTC)
        universe = PointInTimeUniverse.from_tidy(tidy, as_of)
        assert universe.symbols == ["A", "B"]
        assert len(universe["A"]) == 5
        assert universe.as_of == as_of
        assert "C" not in universe
        assert universe.get("C") is None

    def test_unknown_symbol_raises_keyerror(self, frame):
        universe = PointInTimeUniverse.from_tidy(
            frame.assign(symbol="A"), dt.datetime(2020, 1, 5, 21, tzinfo=UTC)
        )
        with pytest.raises(KeyError):
            universe["ZZZ"]

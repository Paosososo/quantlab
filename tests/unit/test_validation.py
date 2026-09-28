"""Schema gates and data-quality checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.db.enums import CheckStatus
from quantlab.exceptions import DuplicateObservationError, SchemaValidationError
from quantlab.ingestion.validation import (
    assert_unique_key,
    check_availability_not_before_ts,
    check_calendar_gaps,
    check_extreme_returns,
    check_missing_values,
    check_monotonic_time,
    check_no_duplicates,
    check_ohlc_consistency,
    check_positive_prices,
    check_row_count,
    deduplicate,
    run_price_quality_checks,
    validate_schema,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def good_prices() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "symbol": "A",
            "ts": index,
            "available_at": index + pd.Timedelta(hours=21),
            "open": np.linspace(100, 109, 10),
            "high": np.linspace(101, 110, 10),
            "low": np.linspace(99, 108, 10),
            "close": np.linspace(100.5, 109.5, 10),
            "adj_close": np.linspace(100.5, 109.5, 10),
            "volume": np.full(10, 1e6),
        }
    )


class TestSchemaGate:
    def test_missing_columns_raise(self, good_prices):
        with pytest.raises(SchemaValidationError, match="missing required columns"):
            validate_schema(good_prices.drop(columns=["close"]), required=["close"])

    def test_unparseable_numbers_raise(self, good_prices):
        broken = good_prices.copy()
        broken["close"] = broken["close"].astype(object)
        broken.loc[3, "close"] = "not-a-number"
        with pytest.raises(SchemaValidationError, match="not numeric"):
            validate_schema(broken, required=["close"], numeric=["close"])

    def test_existing_nulls_are_not_counted_as_unparseable(self, good_prices):
        with_null = good_prices.copy()
        with_null.loc[3, "volume"] = np.nan
        out = validate_schema(with_null, required=["volume"], numeric=["volume"])
        assert out["volume"].isna().sum() == 1

    def test_datetime_columns_are_localised(self, good_prices):
        naive = good_prices.copy()
        naive["ts"] = naive["ts"].dt.tz_localize(None)
        out = validate_schema(naive, required=["ts"], datetime_utc=["ts"])
        assert str(out["ts"].dt.tz) == "UTC"

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_unparseable_dates_raise(self, good_prices):
        broken = good_prices.copy()
        broken["ts"] = "not-a-date"
        with pytest.raises(SchemaValidationError, match="UTC timestamp"):
            validate_schema(broken, required=["ts"], datetime_utc=["ts"])


class TestDuplicates:
    def test_assert_unique_key_raises_on_duplicates(self, good_prices):
        doubled = pd.concat([good_prices, good_prices.iloc[[0]]], ignore_index=True)
        with pytest.raises(DuplicateObservationError):
            assert_unique_key(doubled, ["symbol", "ts"])

    def test_deduplicate_keeps_the_last_observation(self):
        """Providers append corrections, so the later row is the corrected one."""
        frame = pd.DataFrame({"k": [1, 1, 2], "v": [10.0, 11.0, 20.0]})
        out, dropped = deduplicate(frame, ["k"])
        assert dropped == 1
        assert out.loc[out["k"] == 1, "v"].item() == 11.0

    def test_deduplicate_is_a_no_op_on_clean_data(self, good_prices):
        out, dropped = deduplicate(good_prices, ["symbol", "ts"])
        assert dropped == 0
        assert len(out) == len(good_prices)


class TestQualityChecks:
    def test_clean_data_passes_everything(self, good_prices):
        report = run_price_quality_checks(good_prices, entity="A")
        assert report.ok
        assert not report.failures

    def test_duplicate_rows_fail(self, good_prices):
        doubled = pd.concat([good_prices, good_prices.iloc[[0]]], ignore_index=True)
        assert check_no_duplicates(doubled, ["symbol", "ts"]).status is CheckStatus.FAILED

    def test_unsorted_timestamps_fail(self, good_prices):
        shuffled = good_prices.iloc[::-1].reset_index(drop=True)
        assert check_monotonic_time(shuffled).status is CheckStatus.FAILED

    def test_availability_before_observation_fails(self, good_prices):
        broken = good_prices.copy()
        broken["available_at"] = broken["ts"] - pd.Timedelta(days=1)
        assert check_availability_not_before_ts(broken).status is CheckStatus.FAILED

    def test_impossible_ohlc_fails(self, good_prices):
        broken = good_prices.copy()
        broken.loc[2, "high"] = broken.loc[2, "low"] - 1.0
        assert check_ohlc_consistency(broken).status is CheckStatus.FAILED

    def test_non_positive_price_fails(self, good_prices):
        broken = good_prices.copy()
        broken.loc[4, "close"] = 0.0
        assert check_positive_prices(broken, ("close",)).status is CheckStatus.FAILED

    def test_missing_values_warn_rather_than_fail(self, good_prices):
        broken = good_prices.copy()
        broken.loc[0:4, "adj_close"] = np.nan
        result = check_missing_values(broken, ("adj_close",))
        assert result.status is CheckStatus.WARNED

    def test_extreme_moves_warn(self, good_prices):
        broken = good_prices.copy()
        broken.loc[5, "adj_close"] = broken.loc[5, "adj_close"] * 3
        assert check_extreme_returns(broken).status is CheckStatus.WARNED

    def test_calendar_gaps_warn(self):
        index = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-03-01"], tz="UTC")
        frame = pd.DataFrame({"ts": index})
        assert check_calendar_gaps(frame).status is CheckStatus.WARNED

    def test_weekend_gaps_do_not_warn(self, good_prices):
        assert check_calendar_gaps(good_prices).status is CheckStatus.PASSED

    def test_empty_frame_fails_the_row_count_check(self):
        assert check_row_count(pd.DataFrame(), minimum=1).status is CheckStatus.FAILED

    def test_report_rows_are_persistable(self, good_prices):
        report = run_price_quality_checks(good_prices, entity="A")
        rows = report.to_rows(ingestion_run_id=7)
        assert len(rows) == len(report.results)
        assert {"dataset", "check_name", "status", "checked_at"} <= set(rows[0])
        assert rows[0]["ingestion_run_id"] == 7

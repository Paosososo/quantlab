"""Provider parsers, tested against checked-in fixture bytes.

Parsing is where provider quirks bite: header renames, '.' for missing values,
locale-specific dates.  These tests are pure and offline, so a parser regression
is caught in milliseconds without depending on a public endpoint being up.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quantlab.db.enums import Frequency
from quantlab.exceptions import ProviderError
from quantlab.ingestion.base import ECONOMIC_COLUMNS, PRICE_COLUMNS, RawPayload
from quantlab.ingestion.providers.fred import KNOWN_SERIES, FredProvider
from quantlab.ingestion.providers.stooq import StooqProvider
from quantlab.ingestion.providers.synthetic import SyntheticProvider, SyntheticSpec
from quantlab.ingestion.registry import (
    available_price_providers,
    get_economic_provider,
    get_price_provider,
)
from quantlab.timeutils import UTC
from tests.conftest import FIXTURES

pytestmark = pytest.mark.unit


def payload(name: str, provider: str, dataset: str, entity: str) -> RawPayload:
    return RawPayload(
        provider=provider,
        dataset=dataset,
        entity=entity,
        content=(FIXTURES / name).read_bytes(),
        content_type="text/csv",
        fetched_at=dt.datetime(2024, 1, 1, tzinfo=UTC),
    )


class TestStooqParser:
    @pytest.fixture
    def parsed(self) -> pd.DataFrame:
        raw = payload("stooq_spy_sample.csv", "stooq", "daily_bars", "spy.us")
        return StooqProvider().parse(raw)

    def test_columns_match_the_canonical_schema(self, parsed):
        assert list(parsed.columns) == list(PRICE_COLUMNS)

    def test_values_survive_the_round_trip(self, parsed):
        first = parsed.iloc[0]
        assert first["symbol"] == "SPY.US"
        assert first["open"] == pytest.approx(323.54)
        assert first["close"] == pytest.approx(324.87)
        assert first["volume"] == pytest.approx(59_151_200)

    def test_availability_is_the_us_session_close(self, parsed):
        """2 January 2020 is winter, so 16:00 New York is 21:00 UTC."""
        assert parsed.iloc[0]["available_at"] == pd.Timestamp("2020-01-02 21:00", tz="UTC")

    def test_availability_is_never_before_the_bar(self, parsed):
        assert (parsed["available_at"] >= parsed["ts"]).all()

    def test_rows_are_sorted(self, parsed):
        assert parsed["ts"].is_monotonic_increasing

    def test_adjusted_close_mirrors_close(self, parsed):
        """Documented limitation: Stooq gives one close series, so we do not
        pretend to have a dividend-adjusted one."""
        assert (parsed["adj_close"] == parsed["close"]).all()

    def test_a_non_csv_body_raises(self):
        raw = RawPayload(
            provider="stooq",
            dataset="daily_bars",
            entity="bogus",
            content=b"Exceeded the daily hits limit",
            content_type="text/csv",
            fetched_at=dt.datetime(2024, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(ProviderError):
            StooqProvider().parse(raw)

    def test_session_is_chosen_from_the_ticker_suffix(self):
        provider = StooqProvider()
        assert provider.session_for("spy.us").name == "us_equity"
        assert provider.session_for("vod.uk").name == "lse"
        assert provider.session_for("^spx").name == "us_equity"


class TestFredParser:
    def test_missing_values_marked_with_a_dot_become_nan(self):
        raw = payload("fred_dgs10_sample.csv", "fred", "economic_series", "DGS10")
        provider = FredProvider()
        parsed = provider.parse(raw, provider.metadata_for("DGS10"))
        assert list(parsed.columns) == list(ECONOMIC_COLUMNS)
        assert np.isnan(parsed.iloc[0]["value"])
        assert parsed.iloc[1]["value"] == pytest.approx(1.88)

    def test_the_legacy_date_header_is_handled(self):
        """FRED renamed the first column from DATE to observation_date."""
        raw = payload("fred_legacy_header_sample.csv", "fred", "economic_series", "UNRATE")
        provider = FredProvider()
        parsed = provider.parse(raw, provider.metadata_for("UNRATE"))
        assert len(parsed) == 3
        assert parsed.iloc[0]["value"] == pytest.approx(3.6)

    def test_the_publication_lag_pushes_availability_out(self):
        raw = payload("fred_legacy_header_sample.csv", "fred", "economic_series", "UNRATE")
        provider = FredProvider()
        metadata = provider.metadata_for("UNRATE")
        parsed = provider.parse(raw, metadata)
        gap = parsed.iloc[0]["available_at"] - parsed.iloc[0]["ts"]
        assert gap.days == metadata.publication_lag_days
        assert metadata.publication_lag_days >= 30  # a monthly release is never same-day

    def test_daily_series_have_short_lags_and_monthly_ones_long(self):
        assert KNOWN_SERIES["DGS10"].publication_lag_days <= 2
        assert KNOWN_SERIES["CPIAUCSL"].publication_lag_days >= 30
        assert KNOWN_SERIES["GDPC1"].frequency is Frequency.QUARTERLY

    def test_an_unknown_series_gets_the_conservative_default(self):
        metadata = FredProvider().metadata_for("MADE_UP_SERIES")
        assert metadata.publication_lag_days > 0

    def test_a_body_with_one_column_raises(self):
        raw = RawPayload(
            provider="fred",
            dataset="economic_series",
            entity="X",
            content=b"only_one_column\n1\n2\n",
            content_type="text/csv",
            fetched_at=dt.datetime(2024, 1, 1, tzinfo=UTC),
        )
        provider = FredProvider()
        with pytest.raises(ProviderError):
            provider.parse(raw, provider.metadata_for("X"))


class TestSyntheticProvider:
    def test_generation_is_deterministic_for_a_seed(self, short_spec):
        import asyncio

        a = asyncio.run(SyntheticProvider(short_spec, seed=5).fetch("SYN_A"))
        b = asyncio.run(SyntheticProvider(short_spec, seed=5).fetch("SYN_A"))
        assert a.sha256 == b.sha256

    def test_different_seeds_give_different_paths(self, short_spec):
        import asyncio

        a = asyncio.run(SyntheticProvider(short_spec, seed=5).fetch("SYN_A"))
        b = asyncio.run(SyntheticProvider(short_spec, seed=6).fetch("SYN_A"))
        assert a.sha256 != b.sha256

    def test_different_symbols_give_different_paths(self, short_spec):
        import asyncio

        a = asyncio.run(SyntheticProvider(short_spec, seed=5).fetch("SYN_A"))
        b = asyncio.run(SyntheticProvider(short_spec, seed=5).fetch("SYN_B"))
        assert a.sha256 != b.sha256

    def test_ohlc_invariants_hold(self, price_frame):
        body_high = price_frame[["open", "close"]].max(axis=1)
        body_low = price_frame[["open", "close"]].min(axis=1)
        assert (price_frame["high"] >= body_high - 1e-9).all()
        assert (price_frame["low"] <= body_low + 1e-9).all()
        assert (price_frame["volume"] >= 0).all()

    def test_the_open_differs_from_the_previous_close(self, price_frame):
        """Overnight gaps are what make next-open execution a real test."""
        gaps = price_frame["open"].iloc[1:].to_numpy() - price_frame["close"].iloc[:-1].to_numpy()
        assert np.abs(gaps).mean() > 0

    def test_injected_autocorrelation_is_recoverable(self):
        """The generator's control knob works, which is what lets a test assert
        that the research pipeline finds signal when signal exists."""
        from quantlab.ingestion.providers.synthetic import generate_price_frame

        spec = SyntheticSpec(
            start=dt.date(2000, 1, 1),
            end=dt.date(2020, 12, 31),
            autocorrelation=0.3,
            regime_switching=False,
        )
        frame = generate_price_frame("SYN_AR", spec, seed=1)
        returns = np.diff(np.log(frame["adj_close"].to_numpy()))
        rho = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        assert rho == pytest.approx(0.3, abs=0.06)

    def test_zero_autocorrelation_gives_a_random_walk(self):
        from quantlab.ingestion.providers.synthetic import generate_price_frame

        spec = SyntheticSpec(
            start=dt.date(2000, 1, 1),
            end=dt.date(2020, 12, 31),
            autocorrelation=0.0,
            regime_switching=False,
        )
        frame = generate_price_frame("SYN_RW", spec, seed=2)
        returns = np.diff(np.log(frame["adj_close"].to_numpy()))
        rho = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        assert abs(rho) < 0.05

    def test_the_spec_is_recorded_in_the_payload(self, short_spec):
        import asyncio

        raw = asyncio.run(SyntheticProvider(short_spec, seed=5).fetch("SYN_A"))
        assert raw.request["generator"] == "synthetic"
        assert "annual_volatility" in raw.request["spec"]


class TestRegistry:
    def test_known_providers_are_listed(self):
        assert {"stooq", "synthetic"} <= set(available_price_providers())

    def test_lookup_returns_an_instance(self):
        assert get_price_provider("synthetic").name == "synthetic"
        assert get_economic_provider("fred").name == "fred"

    def test_an_unknown_provider_raises_with_the_known_list(self):
        with pytest.raises(ProviderError, match="unknown price provider"):
            get_price_provider("bloomberg")

"""End-to-end ingestion against the synthetic provider.

Uses the real pipeline, the real validation gate and the real database writer.
The only thing replaced is the network, and it is replaced by a provider that
implements the same interface, so these tests exercise the production code path
rather than a mock of it.
"""

from __future__ import annotations

import datetime as dt

import pytest

from quantlab.db import models as m
from quantlab.db import repository as repo
from quantlab.db.enums import AssetClass, CheckStatus, ReturnDirection, RunStatus
from quantlab.exceptions import ProviderError
from quantlab.ingestion.base import RawPayload
from quantlab.ingestion.pipeline import ingest_prices, materialise_returns
from quantlab.ingestion.providers.synthetic import SyntheticProvider, SyntheticSpec
from quantlab.ingestion.raw_store import RawStore

pytestmark = pytest.mark.integration

SPEC = SyntheticSpec(start=dt.date(2020, 1, 1), end=dt.date(2020, 12, 31))


class TestPriceIngestion:
    def test_a_first_load_populates_assets_prices_and_the_run_log(self, session, tmp_path):
        summary = ingest_prices(
            session,
            SyntheticProvider(SPEC, seed=1),
            ["SYN_A", "SYN_B"],
            asset_class=AssetClass.EQUITY,
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()

        assert not summary.failed
        assert summary.rows_written > 400
        assert repo.count_rows(session, m.Asset) == 2
        assert repo.count_rows(session, m.Price) == summary.rows_written
        runs = session.query(m.IngestionRun).all()
        assert len(runs) == 2
        assert all(r.status is RunStatus.SUCCEEDED for r in runs)
        assert all(r.watermark_to is not None for r in runs)

    def test_the_raw_payload_is_archived_with_a_manifest(self, session, tmp_path):
        store = RawStore(tmp_path / "raw")
        ingest_prices(session, SyntheticProvider(SPEC, seed=1), ["SYN_A"], raw_store=store)
        session.flush()
        files = sorted(p.name for p in (tmp_path / "raw").rglob("*") if p.is_file())
        assert any(name.endswith(".csv") for name in files)
        assert any(name.endswith(".manifest.json") for name in files)

    def test_the_run_records_the_payload_digest(self, session, tmp_path):
        ingest_prices(
            session,
            SyntheticProvider(SPEC, seed=1),
            ["SYN_A"],
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()
        run = session.query(m.IngestionRun).one()
        assert run.payload_sha256 and len(run.payload_sha256) == 64
        assert run.raw_path

    def test_rerunning_is_idempotent(self, session, tmp_path):
        store = RawStore(tmp_path / "raw")
        ingest_prices(
            session, SyntheticProvider(SPEC, seed=1), ["SYN_A"], start=SPEC.start, raw_store=store
        )
        session.flush()
        first = repo.count_rows(session, m.Price)

        ingest_prices(
            session, SyntheticProvider(SPEC, seed=1), ["SYN_A"], start=SPEC.start, raw_store=store
        )
        session.flush()
        assert repo.count_rows(session, m.Price) == first

    def test_the_second_run_is_incremental(self, session, tmp_path):
        """Without an explicit start, the loader resumes from the watermark."""
        store = RawStore(tmp_path / "raw")
        first = ingest_prices(session, SyntheticProvider(SPEC, seed=1), ["SYN_A"], raw_store=store)
        session.flush()
        second = ingest_prices(session, SyntheticProvider(SPEC, seed=1), ["SYN_A"], raw_store=store)
        session.flush()
        assert second.results[0].rows_fetched < first.results[0].rows_fetched
        assert second.results[0].rows_fetched > 0  # the overlap window is re-read

    def test_quality_checks_are_persisted(self, session, tmp_path):
        ingest_prices(
            session,
            SyntheticProvider(SPEC, seed=1),
            ["SYN_A"],
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()
        checks = session.query(m.DataQualityCheck).all()
        assert checks
        assert {c.status for c in checks} <= {CheckStatus.PASSED, CheckStatus.WARNED}
        assert any(c.check_name == "availability_after_ts" for c in checks)

    def test_one_bad_symbol_does_not_sink_the_run(self, session, tmp_path):
        class PartlyBroken(SyntheticProvider):
            async def fetch(self, symbol, start=None, end=None):
                if symbol == "BROKEN":
                    raise ProviderError("simulated provider outage", symbol=symbol)
                return await super().fetch(symbol, start, end)

        summary = ingest_prices(
            session,
            PartlyBroken(SPEC, seed=1),
            ["SYN_A", "BROKEN"],
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()
        assert len(summary.succeeded) == 1
        assert len(summary.failed) == 1
        assert "simulated provider outage" in summary.failed[0].error
        summary.raise_if_all_failed()  # partial failure is tolerated

    def test_total_failure_is_escalated(self, session, tmp_path):
        class Broken(SyntheticProvider):
            async def fetch(self, symbol, start=None, end=None):
                raise ProviderError("everything is down", symbol=symbol)

        summary = ingest_prices(
            session, Broken(SPEC, seed=1), ["A", "B"], raw_store=RawStore(tmp_path / "raw")
        )
        session.flush()
        with pytest.raises(Exception, match="every entity failed"):
            summary.raise_if_all_failed()

    def test_data_failing_the_quality_gate_is_not_loaded(self, session, tmp_path):
        """A provider that returns duplicated bars must not reach the database."""

        class DuplicatingProvider(SyntheticProvider):
            def parse(self, payload: RawPayload):
                frame = super().parse(payload)
                frame.loc[5, "high"] = 0.0  # breaks the OHLC invariant
                return frame

        summary = ingest_prices(
            session,
            DuplicatingProvider(SPEC, seed=1),
            ["SYN_A"],
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()
        assert summary.failed
        assert repo.count_rows(session, m.Price) == 0
        failed_checks = (
            session.query(m.DataQualityCheck)
            .filter(m.DataQualityCheck.status == CheckStatus.FAILED)
            .all()
        )
        assert failed_checks


class TestDerivedReturns:
    def test_forward_returns_become_available_only_after_the_horizon(self, session, tmp_path):
        ingest_prices(
            session,
            SyntheticProvider(SPEC, seed=1),
            ["SYN_A"],
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()
        materialise_returns(session, ["SYN_A"], horizons=(5,))
        session.flush()

        forward = repo.load_returns(
            session, ["SYN_A"], horizon_days=5, direction=ReturnDirection.FORWARD
        )
        trailing = repo.load_returns(
            session, ["SYN_A"], horizon_days=5, direction=ReturnDirection.TRAILING
        )
        assert not forward.empty
        assert (forward["available_at"] > forward["ts"]).all()
        # A forward return is knowable strictly later than a trailing one for
        # the same date; that gap is what stops a label leaking into features.
        merged = forward.merge(trailing, on="ts", suffixes=("_fwd", "_trl"))
        assert (merged["available_at_fwd"] > merged["available_at_trl"]).all()

    def test_as_of_filtering_hides_unrealised_labels(self, session, tmp_path):
        ingest_prices(
            session,
            SyntheticProvider(SPEC, seed=1),
            ["SYN_A"],
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()
        materialise_returns(session, ["SYN_A"], horizons=(21,))
        session.flush()

        cutoff = dt.datetime(2020, 6, 30, tzinfo=dt.timezone.utc)
        visible = repo.load_returns(
            session, ["SYN_A"], horizon_days=21, direction=ReturnDirection.FORWARD, as_of=cutoff
        )
        assert not visible.empty
        assert visible["available_at"].max() <= cutoff
        # The most recent 21 sessions have no knowable label yet.
        assert visible["ts"].max() < cutoff


class TestPriceLoading:
    def test_as_of_filtering_hides_unpublished_bars(self, session, tmp_path):
        ingest_prices(
            session,
            SyntheticProvider(SPEC, seed=1),
            ["SYN_A"],
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()
        cutoff = dt.datetime(2020, 3, 16, 12, tzinfo=dt.timezone.utc)
        visible = repo.load_prices(session, ["SYN_A"], as_of=cutoff)
        assert visible["available_at"].max() <= cutoff

    def test_loaded_prices_are_floats_not_decimals(self, session, tmp_path):
        ingest_prices(
            session,
            SyntheticProvider(SPEC, seed=1),
            ["SYN_A"],
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()
        frame = repo.load_prices(session, ["SYN_A"])
        assert frame["close"].dtype == "float64"
        assert str(frame["ts"].dt.tz) == "UTC"

    def test_wide_frames_pivot_on_symbol(self, session, tmp_path):
        ingest_prices(
            session,
            SyntheticProvider(SPEC, seed=1),
            ["SYN_A", "SYN_B"],
            raw_store=RawStore(tmp_path / "raw"),
        )
        session.flush()
        wide = repo.wide_close_frame(repo.load_prices(session, ["SYN_A", "SYN_B"]))
        assert list(wide.columns) == ["SYN_A", "SYN_B"]
        assert wide.index.is_monotonic_increasing

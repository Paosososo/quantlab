"""The orchestration layer.

These are the functions Airflow tasks and CLI commands call, so a bug here fails
in production rather than in a notebook.  Everything runs against the synthetic
provider and a temporary SQLite database, so the suite stays offline.
"""

from __future__ import annotations

import datetime as dt

import pytest

from quantlab.db import models as m
from quantlab.db import repository as repo
from quantlab.db.enums import AssetClass, CheckStatus
from quantlab.db.session import get_sessionmaker
from quantlab.exceptions import DataQualityError, QuantlabError
from quantlab.ingestion.providers.synthetic import SyntheticSpec
from quantlab.pipelines import (
    DEFAULT_MACRO_CODES,
    DEFAULT_SYMBOLS,
    run_backtests,
    run_data_validation,
    run_feature_generation,
    run_model_training,
    run_price_ingestion,
    run_return_materialisation,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SYMBOLS = ["SYN_P", "SYN_Q"]
SPEC = SyntheticSpec(start=dt.date(2015, 1, 1), end=dt.date(2021, 12, 31))


@pytest.fixture
def wired(engine, monkeypatch, tmp_path):
    """Point the process-wide engine at the test database.

    ``quantlab.pipelines`` deliberately opens its own sessions, because an
    Airflow task has no session to inherit.  That means the test has to redirect
    the singleton rather than inject a session, which is exactly what a Docker
    deployment does through the environment.
    """
    from quantlab.db import session as session_module

    monkeypatch.setattr(session_module, "get_engine", lambda: engine)
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(tmp_path / "data"))
    return engine


def ingest(**overrides):
    kwargs = {
        "provider": "synthetic",
        "asset_class": AssetClass.ETF,
        "spec": SPEC,
        "seed": 11,
    }
    kwargs.update(overrides)
    return run_price_ingestion(SYMBOLS, **kwargs)


class TestDefaults:
    def test_the_demo_universe_is_liquid_etfs(self):
        assert len(DEFAULT_SYMBOLS) >= 5
        assert all(s.endswith(".us") for s in DEFAULT_SYMBOLS)

    def test_macro_defaults_cover_rates_credit_and_activity(self):
        assert {"DGS10", "T10Y2Y", "VIXCLS"} <= set(DEFAULT_MACRO_CODES)


class TestPriceIngestion:
    def test_prices_are_loaded_and_reported(self, wired):
        summary = ingest()
        assert set(summary["succeeded"]) == set(SYMBOLS)
        assert not summary["failed"]
        assert summary["rows_written"] > 1_000

        with get_sessionmaker(wired)() as session:
            assert repo.count_rows(session, m.Asset) == 2
            assert repo.count_rows(session, m.Price) == summary["rows_written"]

    def test_rerunning_does_not_duplicate(self, wired):
        ingest(start=SPEC.start)
        with get_sessionmaker(wired)() as session:
            first = repo.count_rows(session, m.Price)
        ingest(start=SPEC.start)
        with get_sessionmaker(wired)() as session:
            assert repo.count_rows(session, m.Price) == first

    def test_an_unknown_provider_is_rejected(self, wired):
        with pytest.raises(QuantlabError, match="unknown price provider"):
            run_price_ingestion(SYMBOLS, provider="bloomberg")

    def test_fail_on_any_error_escalates_a_partial_failure(self, wired, monkeypatch):
        from quantlab.ingestion.providers.synthetic import SyntheticProvider

        original = SyntheticProvider.fetch

        async def flaky(self, symbol, start=None, end=None):
            if symbol == "SYN_Q":
                raise RuntimeError("simulated outage")
            return await original(self, symbol, start, end)

        monkeypatch.setattr(SyntheticProvider, "fetch", flaky)
        with pytest.raises(QuantlabError, match="some symbols failed"):
            ingest(fail_on_any_error=True)


class TestReturnsAndValidation:
    def test_returns_are_materialised_for_every_asset(self, wired):
        ingest()
        summary = run_return_materialisation(horizons=(1, 5))
        assert summary["symbols"] == 2
        assert summary["rows_written"] > 1_000

    def test_validation_passes_on_freshly_loaded_data(self, wired):
        ingest()
        # The synthetic sample ends in 2021, so a staleness check against today
        # would warn; the point of this call is the coverage check.
        report = run_data_validation(max_stale_days=100_000, fail_on_error=True)
        assert set(report["checked"]) == set(SYMBOLS)
        assert not report["issues"]

    def test_validation_fails_for_an_asset_with_no_prices(self, wired):
        ingest()
        with get_sessionmaker(wired)() as session:
            repo.get_or_create_asset(session, "EMPTY", asset_class=AssetClass.EQUITY)
            session.commit()
        with pytest.raises(DataQualityError):
            run_data_validation(max_stale_days=100_000, fail_on_error=True)

    def test_staleness_is_a_warning_not_a_failure(self, wired):
        ingest()
        report = run_data_validation(max_stale_days=1, fail_on_error=True)
        assert report["issues"]
        assert all(i["status"] == str(CheckStatus.WARNED) for i in report["issues"])

    def test_checks_are_persisted_for_later_querying(self, wired):
        ingest()
        run_data_validation(max_stale_days=100_000)
        with get_sessionmaker(wired)() as session:
            rows = (
                session.query(m.DataQualityCheck)
                .filter(m.DataQualityCheck.check_name == "warehouse_coverage")
                .all()
            )
            assert len(rows) == 2


class TestFeatureGeneration:
    def test_features_are_built_and_persisted(self, wired):
        ingest()
        summary = run_feature_generation(include_macro=False)
        assert summary["n_features"] > 15
        assert summary["values_written"] > 10_000
        assert len(summary["spec_hash"]) == 64
        with get_sessionmaker(wired)() as session:
            assert repo.count_rows(session, m.FeatureSet) == 1

    def test_rebuilding_is_idempotent(self, wired):
        ingest()
        first = run_feature_generation(include_macro=False)
        second = run_feature_generation(include_macro=False)
        assert first["spec_hash"] == second["spec_hash"]
        with get_sessionmaker(wired)() as session:
            assert repo.count_rows(session, m.FeatureSet) == 1
            assert repo.count_rows(session, m.FeatureValue) == first["values_written"]

    def test_coverage_is_reported_worst_first(self, wired):
        ingest()
        summary = run_feature_generation(include_macro=False)
        coverage = summary["worst_coverage"]
        assert coverage
        assert coverage[0]["coverage"] <= coverage[-1]["coverage"]

    def test_no_prices_is_an_explicit_error(self, wired):
        with pytest.raises(QuantlabError, match="no prices"):
            run_feature_generation(include_macro=False)


class TestModelTraining:
    def test_the_ladder_trains_and_persists(self, wired):
        ingest()
        summary = run_model_training(
            model_specs=[("zero", {}), ("ridge", {"alpha": 10.0})],
            include_macro=False,
            n_splits=2,
            test_size=126,
            min_train_size=504,
            run_prefix="test",
        )
        assert summary["n_models"] == 2
        assert len(summary["data_fingerprint"]) == 64
        with get_sessionmaker(wired)() as session:
            runs = session.query(m.ModelRun).all()
            assert len(runs) == 2
            assert all(r.config_hash for r in runs)
            assert all(r.seed is not None for r in runs)
            assert repo.count_rows(session, m.Prediction) > 100

    def test_the_same_configuration_hashes_the_same(self, wired):
        ingest()
        kwargs = {
            "model_specs": [("ridge", {"alpha": 10.0})],
            "include_macro": False,
            "n_splits": 2,
            "test_size": 126,
            "min_train_size": 504,
            "persist": False,
        }
        first = run_model_training(**kwargs)
        second = run_model_training(**kwargs)
        assert first["results"][0]["config_hash"] == second["results"][0]["config_hash"]
        assert first["data_fingerprint"] == second["data_fingerprint"]

    def test_metrics_are_attached_to_each_result(self, wired):
        ingest()
        summary = run_model_training(
            model_specs=[("zero", {})],
            include_macro=False,
            n_splits=2,
            test_size=126,
            min_train_size=504,
            persist=False,
        )
        metrics = summary["results"][0]["metrics"]
        assert metrics["n"] > 100
        assert metrics["r2_oos_vs_zero"] == pytest.approx(0.0)


class TestBacktests:
    def test_the_classical_set_runs_and_persists(self, wired):
        ingest()
        summary = run_backtests(benchmark="SYN_P", rebalance="monthly")
        assert summary["n_backtests"] == 3
        names = {r["strategy"] for r in summary["results"]}
        assert names == {"buy_and_hold", "ma_crossover", "ts_momentum"}
        with get_sessionmaker(wired)() as session:
            assert repo.count_rows(session, m.Backtest) == 3
            assert repo.count_rows(session, m.PortfolioSnapshot) > 1_000
            assert repo.count_rows(session, m.Trade) > 0

    def test_every_persisted_trade_respects_the_temporal_contract(self, wired):
        ingest()
        run_backtests(benchmark="SYN_P", rebalance="monthly")
        with get_sessionmaker(wired)() as session:
            trades = session.query(m.Trade).all()
            assert trades
            assert all(t.execution_ts > t.decision_ts for t in trades)

    def test_metrics_are_recorded_on_the_backtest_row(self, wired):
        ingest()
        run_backtests(benchmark="SYN_P", rebalance="monthly")
        with get_sessionmaker(wired)() as session:
            for row in session.query(m.Backtest).all():
                assert row.metrics["periods"] > 1_000
                assert "sharpe_ratio" in row.metrics
                assert row.config_hash

    def test_no_prices_is_an_explicit_error(self, wired):
        with pytest.raises(QuantlabError, match="no prices"):
            run_backtests()

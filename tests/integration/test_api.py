"""API tests against a seeded database.

Uses FastAPI's TestClient with a real (SQLite) database populated by the real
ingestion and backtesting pipelines.  Nothing is mocked, so a broken query or a
bad response model fails here rather than in production.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quantlab.backtesting.costs import CostModel, FixedBpsSlippage, PerShareCommission
from quantlab.backtesting.engine import BacktestEngine, EngineConfig
from quantlab.backtesting.execution import ExecutionModel
from quantlab.backtesting.market import MarketData
from quantlab.backtesting.results import build_result, persist_result
from quantlab.backtesting.sizing import TargetWeightSizer
from quantlab.backtesting.strategy import BuyAndHold, MovingAverageCrossover
from quantlab.db import repository as repo
from quantlab.db.enums import AssetClass
from quantlab.db.session import get_sessionmaker
from quantlab.features.library import default_feature_set
from quantlab.features.pipeline import (
    build_feature_frame,
    persist_feature_set,
    persist_feature_values,
)
from quantlab.ingestion.pipeline import ingest_prices, materialise_returns
from quantlab.ingestion.providers.synthetic import SyntheticSpec
from quantlab.ingestion.raw_store import RawStore

pytestmark = pytest.mark.integration

SPEC = SyntheticSpec(start=dt.date(2019, 1, 1), end=dt.date(2021, 12, 31))
SYMBOLS = ("SYN_A", "SYN_B")


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory) -> Path:
    """A database seeded once for the whole module.

    Module-scoped on purpose.  Seeding runs the real ingestion, feature and
    backtest pipelines, which takes several seconds; doing that once instead of
    once per test turns a four-minute suite into a five-second one.  It is safe
    because every endpoint under test is read-only.
    """
    from quantlab.db.base import Base
    from quantlab.db.session import build_engine
    from quantlab.ingestion.providers.synthetic import SyntheticProvider

    root = tmp_path_factory.mktemp("api")
    db_path = root / "api.db"
    engine = build_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = get_sessionmaker(engine)

    with factory() as session:
        ingest_prices(
            session,
            SyntheticProvider(SPEC, seed=3),
            list(SYMBOLS),
            asset_class=AssetClass.ETF,
            raw_store=RawStore(root / "raw"),
        )
        session.commit()

        materialise_returns(session, list(SYMBOLS), horizons=(1, 5))
        session.commit()

        prices = repo.load_prices(session, list(SYMBOLS))
        feature_set = default_feature_set(include_macro=False)
        frame = build_feature_frame(prices, feature_set)
        row = persist_feature_set(session, feature_set)
        persist_feature_values(session, row, frame, feature_names=feature_set.names)
        session.commit()

        market = MarketData(prices)
        config = EngineConfig(
            initial_cash=1_000_000.0,
            execution=ExecutionModel(costs=CostModel(PerShareCommission(), FixedBpsSlippage(5.0))),
            sizer=TargetWeightSizer(allow_fractional=False),
            rebalance="monthly",
            benchmark_symbol="SYN_A",
        )
        for strategy in (
            BuyAndHold(list(SYMBOLS)),
            MovingAverageCrossover(list(SYMBOLS), fast=20, slow=60),
        ):
            output = BacktestEngine(config).run(strategy, market)
            persist_result(
                session,
                build_result(output),
                strategy_name=strategy.name,
                strategy_kind=type(strategy).__name__,
                strategy_params=strategy.describe(),
                benchmark_symbol="SYN_A",
            )
        session.commit()

    engine.dispose()
    return db_path


@pytest.fixture
def client(seeded_db: Path, monkeypatch) -> Iterator[TestClient]:
    """A client wired to the seeded database through the real settings path.

    Pointing ``QUANTLAB_DATABASE_URL`` at the seeded file rather than
    monkeypatching ``get_engine`` means the test exercises the actual dependency
    chain -- settings, engine factory, session dependency -- instead of a
    stand-in for it.
    """
    from quantlab.api.main import create_app
    from quantlab.config import reset_settings
    from quantlab.db.session import reset_engine

    monkeypatch.setenv("QUANTLAB_DATABASE_URL", f"sqlite:///{seeded_db}")
    reset_settings()
    reset_engine()
    try:
        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        reset_engine()
        reset_settings()


class TestHealth:
    def test_health_reports_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["version"]

    def test_root_lists_entry_points(self, client):
        payload = client.get("/").json()
        assert payload["docs"] == "/docs"

    def test_openapi_schema_is_valid(self, client):
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "quantlab"
        assert len(schema["paths"]) > 15

    def test_responses_carry_a_timing_header(self, client):
        assert "X-Response-Time-Ms" in client.get("/health").headers


class TestAssets:
    def test_listing_returns_the_seeded_assets(self, client):
        payload = client.get("/assets").json()
        assert payload["total"] == 2
        assert {item["symbol"] for item in payload["items"]} == set(SYMBOLS)

    def test_detail_includes_price_coverage(self, client):
        payload = client.get("/assets/SYN_A").json()
        assert payload["n_prices"] > 500
        assert payload["first_price_ts"] < payload["last_price_ts"]

    def test_symbols_are_case_insensitive(self, client):
        assert client.get("/assets/syn_a").status_code == 200

    def test_unknown_asset_returns_404_with_context(self, client):
        response = client.get("/assets/NOPE")
        assert response.status_code == 404
        payload = response.json()
        assert payload["error"] == "not_found"
        assert payload["context"]["symbol"] == "NOPE"

    def test_search_filters(self, client):
        payload = client.get("/assets", params={"search": "SYN_B"}).json()
        assert [i["symbol"] for i in payload["items"]] == ["SYN_B"]


class TestPrices:
    def test_prices_are_returned_in_order(self, client):
        payload = client.get("/assets/SYN_A/prices", params={"limit": 10}).json()
        timestamps = [item["ts"] for item in payload["items"]]
        assert timestamps == sorted(timestamps)
        assert payload["returned"] == 10

    def test_as_of_hides_unpublished_bars(self, client):
        """The availability model, exposed over HTTP."""
        cutoff = "2019-06-28T21:00:00Z"
        full = client.get("/assets/SYN_A/prices", params={"limit": 10_000}).json()
        limited = client.get(
            "/assets/SYN_A/prices", params={"limit": 10_000, "as_of": cutoff}
        ).json()
        assert limited["total"] < full["total"]
        assert all(item["available_at"] <= cutoff for item in limited["items"])

    def test_date_range_filters(self, client):
        payload = client.get(
            "/assets/SYN_A/prices",
            params={
                "start": "2020-01-01T00:00:00Z",
                "end": "2020-12-31T00:00:00Z",
                "limit": 10_000,
            },
        ).json()
        assert 200 < payload["total"] < 300

    def test_pagination_does_not_overlap(self, client):
        first = client.get("/assets/SYN_A/prices", params={"limit": 5, "offset": 0}).json()
        second = client.get("/assets/SYN_A/prices", params={"limit": 5, "offset": 5}).json()
        assert {i["ts"] for i in first["items"]}.isdisjoint({i["ts"] for i in second["items"]})

    def test_an_absurd_limit_is_rejected(self, client):
        assert client.get("/assets/SYN_A/prices", params={"limit": 999_999}).status_code == 422


class TestReturns:
    def test_forward_returns_have_a_later_availability(self, client):
        payload = client.get(
            "/assets/SYN_A/returns",
            params={"horizon_days": 5, "direction": "forward", "limit": 20},
        ).json()
        assert payload["items"]
        for item in payload["items"]:
            assert item["available_at"] > item["ts"]

    def test_an_unknown_direction_is_rejected(self, client):
        response = client.get("/assets/SYN_A/returns", params={"direction": "sideways"})
        assert response.status_code == 422


class TestFeatures:
    def test_feature_sets_are_listed_with_their_hash(self, client):
        payload = client.get("/feature-sets").json()
        assert payload
        assert len(payload[0]["spec_hash"]) == 64
        assert payload[0]["n_features"] > 10

    def test_the_definition_is_returned_verbatim(self, client):
        set_id = client.get("/feature-sets").json()[0]["id"]
        payload = client.get(f"/feature-sets/{set_id}").json()
        assert payload["definition"]["features"]
        assert all("lookback" in f for f in payload["definition"]["features"])

    def test_values_can_be_filtered_by_name_and_symbol(self, client):
        set_id = client.get("/feature-sets").json()[0]["id"]
        payload = client.get(
            f"/feature-sets/{set_id}/values",
            params={"symbol": "SYN_A", "names": ["mom_21d"], "limit": 25},
        ).json()
        assert payload["items"]
        assert {i["name"] for i in payload["items"]} == {"mom_21d"}
        assert {i["symbol"] for i in payload["items"]} == {"SYN_A"}

    def test_as_of_applies_to_features_too(self, client):
        set_id = client.get("/feature-sets").json()[0]["id"]
        cutoff = "2019-06-28T21:00:00Z"
        payload = client.get(
            f"/feature-sets/{set_id}/values",
            params={"as_of": cutoff, "limit": 10_000, "names": ["mom_21d"]},
        ).json()
        assert all(i["available_at"] <= cutoff for i in payload["items"])

    def test_unknown_feature_set_is_404(self, client):
        assert client.get("/feature-sets/9999/values").status_code == 404


class TestBacktests:
    def test_both_backtests_are_listed(self, client):
        payload = client.get("/backtests").json()
        assert payload["total"] == 2
        assert {i["strategy_name"] for i in payload["items"]} == {"buy_and_hold", "ma_crossover"}

    def test_metrics_are_attached(self, client):
        item = client.get("/backtests").json()["items"][0]
        assert item["metrics"]["periods"] > 500
        assert "max_drawdown" in item["metrics"]

    def test_the_equity_curve_is_chronological(self, client):
        backtest_id = client.get("/backtests").json()["items"][0]["id"]
        payload = client.get(f"/backtests/{backtest_id}/equity", params={"limit": 10_000}).json()
        timestamps = [p["ts"] for p in payload["items"]]
        assert timestamps == sorted(timestamps)
        assert all(p["drawdown"] <= 1e-9 for p in payload["items"])

    def test_trades_respect_the_temporal_contract(self, client):
        """Every persisted trade must execute strictly after its decision."""
        backtest_id = next(
            i["id"]
            for i in client.get("/backtests").json()["items"]
            if i["strategy_name"] == "ma_crossover"
        )
        payload = client.get(f"/backtests/{backtest_id}/trades", params={"limit": 10_000}).json()
        assert payload["items"]
        for trade in payload["items"]:
            assert trade["execution_ts"] > trade["decision_ts"]

    def test_performance_includes_risk_and_sharpe_inference(self, client):
        backtest_id = client.get("/backtests").json()["items"][0]["id"]
        payload = client.get(f"/backtests/{backtest_id}/performance").json()
        assert payload["risk"]["var_95_historical"] is not None
        inference = payload["sharpe_inference"]
        assert 0.0 <= inference["probabilistic_sharpe"] <= 1.0
        # Null means "no track record length would make this significant", which
        # is the honest answer for a strategy whose Sharpe is at or below the
        # benchmark.  JSON has no infinity, so it serialises as null.
        years = inference["minimum_track_record_years"]
        assert years is None or years > 0
        assert inference["n_observations"] > 100

    def test_deflation_uses_the_stored_trial_count(self, client):
        backtest_id = client.get("/backtests").json()["items"][0]["id"]
        payload = client.get(f"/backtests/{backtest_id}/performance").json()
        inference = payload["sharpe_inference"]
        assert inference["n_trials"] >= 2
        assert inference["trial_source"] == "stored backtests"

    def test_unknown_backtest_is_404(self, client):
        assert client.get("/backtests/9999/performance").status_code == 404


class TestAnalytics:
    def test_minimum_variance_weights_sum_to_one(self, client):
        response = client.post(
            "/analytics/optimise",
            json={"symbols": list(SYMBOLS), "method": "minimum_variance", "lookback_days": 500},
        )
        assert response.status_code == 200
        payload = response.json()
        assert sum(payload["weights"].values()) == pytest.approx(1.0, abs=1e-6)
        assert payload["converged"]

    def test_risk_parity_equalises_contributions(self, client):
        payload = client.post(
            "/analytics/optimise",
            json={"symbols": list(SYMBOLS), "method": "risk_parity", "lookback_days": 500},
        ).json()
        contributions = list(payload["risk_contributions"].values())
        assert max(contributions) - min(contributions) < 0.05

    def test_an_unknown_method_is_rejected(self, client):
        response = client.post(
            "/analytics/optimise", json={"symbols": list(SYMBOLS), "method": "magic"}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "insufficient_data"

    def test_a_single_symbol_fails_validation(self, client):
        response = client.post("/analytics/optimise", json={"symbols": ["SYN_A"]})
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"

    def test_unknown_symbols_return_404(self, client):
        response = client.post("/analytics/optimise", json={"symbols": ["NOPE_A", "NOPE_B"]})
        assert response.status_code == 404

    def test_risk_endpoint_returns_var_and_cvar(self, client):
        payload = client.post(
            "/analytics/risk", json={"symbol": "SYN_A", "lookback_days": 500}
        ).json()
        assert payload["var_95_historical"] < 0
        assert payload["cvar_95"] <= payload["var_95_historical"]

    def test_as_of_changes_the_optimisation_inputs(self, client):
        early = client.post(
            "/analytics/optimise",
            json={
                "symbols": list(SYMBOLS),
                "lookback_days": 400,
                "as_of": "2019-06-28T21:00:00Z",
            },
        ).json()
        late = client.post(
            "/analytics/optimise", json={"symbols": list(SYMBOLS), "lookback_days": 400}
        ).json()
        assert early["weights"] != late["weights"]


class TestModelEndpoints:
    def test_empty_model_runs_return_an_empty_page(self, client):
        payload = client.get("/model-runs").json()
        assert payload["items"] == []
        assert payload["total"] == 0

    def test_comparison_handles_no_runs(self, client):
        assert client.get("/model-runs/comparison").json() == []

    def test_unknown_model_run_is_404(self, client):
        assert client.get("/model-runs/123").status_code == 404

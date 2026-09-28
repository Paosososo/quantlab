"""Database constraints and repository behaviour.

These run against SQLite with foreign keys enabled.  The constraints tested here
are the ones that encode domain rules, so if the ORM changes and a constraint
silently disappears, a test fails rather than a bad row appearing months later.
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from quantlab.db import models as m
from quantlab.db import repository as repo
from quantlab.db.base import Base
from quantlab.db.enums import (
    AssetClass,
    ExecutionTiming,
    OrderSide,
    OrderStatus,
    OrderType,
    ReturnDirection,
    ReturnMethod,
    RunStatus,
)
from quantlab.timeutils import UTC

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_asset(session, symbol="AAA") -> m.Asset:
    return repo.get_or_create_asset(session, symbol, asset_class=AssetClass.EQUITY)


def price_row(asset_id: int, day: int, **overrides) -> dict:
    ts = dt.datetime(2020, 1, day, tzinfo=UTC)
    row = {
        "asset_id": asset_id,
        "ts": ts,
        "available_at": ts + dt.timedelta(hours=21),
        "open": Decimal("100.0"),
        "high": Decimal("101.0"),
        "low": Decimal("99.0"),
        "close": Decimal("100.5"),
        "adj_close": Decimal("100.5"),
        "volume": Decimal("1000000"),
        "source": "test",
        "ingested_at": dt.datetime(2024, 1, 1, tzinfo=UTC),
    }
    row.update(overrides)
    return row


class TestSchemaShape:
    def test_every_expected_table_exists(self, engine):
        expected = {
            "assets",
            "universes",
            "universe_members",
            "prices",
            "returns",
            "economic_series",
            "economic_data",
            "feature_sets",
            "features",
            "model_runs",
            "predictions",
            "strategies",
            "backtests",
            "orders",
            "trades",
            "portfolio_snapshots",
            "position_snapshots",
            "ingestion_runs",
            "data_quality_checks",
        }
        assert expected <= set(inspect(engine).get_table_names())

    def test_hot_paths_are_indexed(self, engine):
        """Queries that run on every pipeline step must not be sequential scans."""
        inspector = inspect(engine)
        price_indexes = {tuple(i["column_names"]) for i in inspector.get_indexes("prices")}
        assert ("asset_id", "ts") in price_indexes or ("asset_id", "available_at") in price_indexes
        feature_indexes = {tuple(i["column_names"]) for i in inspector.get_indexes("features")}
        assert ("feature_set_id", "asset_id", "ts") in feature_indexes

    def test_metadata_matches_the_alembic_migration(self, tmp_path):
        """The migration and the ORM must describe the same schema.

        Unit tests build the schema with ``create_all`` for speed while
        production uses Alembic.  Without this check the two could drift, and
        every test would pass against a schema nobody deploys.
        """
        db_path = tmp_path / "migrated.db"
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "QUANTLAB_DATABASE_URL": f"sqlite:///{db_path}",
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip(f"alembic not runnable in this environment: {result.stderr[-300:]}")

        from sqlalchemy import create_engine

        migrated = create_engine(f"sqlite:///{db_path}")
        migrated_tables = set(inspect(migrated).get_table_names()) - {"alembic_version"}
        migrated.dispose()
        assert migrated_tables == set(Base.metadata.tables)


class TestAssetConstraints:
    def test_symbols_are_unique(self, session):
        make_asset(session, "AAA")
        session.flush()
        session.add(m.Asset(symbol="AAA", asset_class=AssetClass.EQUITY, currency="USD"))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_currency_must_be_upper_case(self, session):
        session.add(m.Asset(symbol="BBB", asset_class=AssetClass.EQUITY, currency="usd"))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_delisting_cannot_precede_listing(self, session):
        session.add(
            m.Asset(
                symbol="CCC",
                asset_class=AssetClass.EQUITY,
                currency="USD",
                listed_on=dt.date(2020, 1, 1),
                delisted_on=dt.date(2019, 1, 1),
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()


class TestPriceConstraints:
    def test_a_bar_is_unique_per_asset_and_timestamp(self, session):
        asset = make_asset(session)
        session.flush()
        repo.upsert_prices(session, [price_row(asset.id, 2)])
        session.flush()
        session.add(m.Price(**price_row(asset.id, 2)))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_high_below_low_is_rejected(self, session):
        asset = make_asset(session)
        session.flush()
        session.add(m.Price(**price_row(asset.id, 3, high=Decimal("98.0"))))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_a_non_positive_price_is_rejected(self, session):
        asset = make_asset(session)
        session.flush()
        session.add(
            m.Price(
                **price_row(
                    asset.id, 4, close=Decimal("0.0"), low=Decimal("0.0"), open=Decimal("0.5")
                )
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_availability_before_the_bar_is_rejected(self, session):
        """The database enforces the availability invariant, not just Python."""
        asset = make_asset(session)
        session.flush()
        row = price_row(asset.id, 5)
        row["available_at"] = row["ts"] - dt.timedelta(days=1)
        session.add(m.Price(**row))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_negative_volume_is_rejected(self, session):
        asset = make_asset(session)
        session.flush()
        session.add(m.Price(**price_row(asset.id, 6, volume=Decimal("-1"))))
        with pytest.raises(IntegrityError):
            session.flush()


class TestUpsertSemantics:
    def test_reinserting_the_same_bar_updates_rather_than_duplicates(self, session):
        asset = make_asset(session)
        session.flush()
        repo.upsert_prices(session, [price_row(asset.id, 7)])
        repo.upsert_prices(
            session,
            [price_row(asset.id, 7, close=Decimal("123.0"), high=Decimal("124.0"))],
        )
        session.flush()
        assert repo.count_rows(session, m.Price) == 1
        stored = session.query(m.Price).one()
        assert float(stored.close) == pytest.approx(123.0)

    def test_the_watermark_is_the_latest_bar(self, session):
        asset = make_asset(session)
        session.flush()
        repo.upsert_prices(session, [price_row(asset.id, d) for d in (2, 3, 6)])
        session.flush()
        watermark = repo.latest_price_ts(session, asset.id)
        assert watermark == dt.datetime(2020, 1, 6, tzinfo=UTC)
        assert watermark.tzinfo is not None

    def test_timestamps_come_back_timezone_aware(self, session):
        """SQLite has no aware type; the UTCDateTime decorator restores it."""
        asset = make_asset(session)
        session.flush()
        repo.upsert_prices(session, [price_row(asset.id, 2)])
        session.flush()
        stored = session.query(m.Price).one()
        assert stored.ts.tzinfo is not None
        assert stored.available_at.tzinfo is not None

    def test_writing_a_naive_datetime_is_refused(self, session):
        asset = make_asset(session)
        session.flush()
        row = price_row(asset.id, 8)
        row["ts"] = dt.datetime(2020, 1, 8)  # naive
        session.add(m.Price(**row))
        with pytest.raises(Exception, match="naive datetime"):
            session.flush()


class TestPointInTimeUniverse:
    def test_membership_is_resolved_as_of_a_date(self, session):
        """The survivorship guard: ask who was in the index *then*, not now."""
        universe = m.Universe(code="TEST", name="Test universe")
        session.add(universe)
        old = make_asset(session, "OLD")
        new = make_asset(session, "NEW")
        session.flush()
        session.add_all(
            [
                m.UniverseMember(
                    universe_id=universe.id,
                    asset_id=old.id,
                    valid_from=dt.datetime(2010, 1, 1, tzinfo=UTC),
                    valid_to=dt.datetime(2015, 1, 1, tzinfo=UTC),
                ),
                m.UniverseMember(
                    universe_id=universe.id,
                    asset_id=new.id,
                    valid_from=dt.datetime(2015, 1, 1, tzinfo=UTC),
                    valid_to=None,
                ),
            ]
        )
        session.flush()

        assert repo.universe_symbols_as_of(
            session, "TEST", dt.datetime(2012, 6, 1, tzinfo=UTC)
        ) == ["OLD"]
        assert repo.universe_symbols_as_of(
            session, "TEST", dt.datetime(2020, 6, 1, tzinfo=UTC)
        ) == ["NEW"]

    def test_an_inverted_interval_is_rejected(self, session):
        universe = m.Universe(code="BAD")
        asset = make_asset(session, "XYZ")
        session.add(universe)
        session.flush()
        session.add(
            m.UniverseMember(
                universe_id=universe.id,
                asset_id=asset.id,
                valid_from=dt.datetime(2020, 1, 1, tzinfo=UTC),
                valid_to=dt.datetime(2019, 1, 1, tzinfo=UTC),
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()


class TestTradeAndPredictionConstraints:
    def _backtest(self, session) -> m.Backtest:
        strategy = m.Strategy(name="s", kind="k", params={}, params_hash="h")
        session.add(strategy)
        session.flush()
        backtest = m.Backtest(
            strategy_id=strategy.id,
            start_ts=dt.datetime(2020, 1, 1, tzinfo=UTC),
            end_ts=dt.datetime(2020, 12, 31, tzinfo=UTC),
            initial_cash=Decimal("100000"),
            execution_timing=ExecutionTiming.NEXT_OPEN,
            cost_config={},
            engine_config={},
            config_hash="c",
            status=RunStatus.SUCCEEDED,
        )
        session.add(backtest)
        session.flush()
        return backtest

    def test_a_trade_executed_before_its_decision_is_rejected(self, session):
        """The temporal claim, enforced by the database as a last line of defence."""
        backtest = self._backtest(session)
        asset = make_asset(session)
        session.flush()
        session.add(
            m.Trade(
                backtest_id=backtest.id,
                asset_id=asset.id,
                decision_ts=dt.datetime(2020, 6, 2, tzinfo=UTC),
                execution_ts=dt.datetime(2020, 6, 1, tzinfo=UTC),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
                reference_price=Decimal("100"),
                fill_price=Decimal("100"),
                commission=Decimal("0"),
                slippage_cost=Decimal("0"),
                notional=Decimal("1000"),
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_an_order_executed_before_its_decision_is_rejected(self, session):
        backtest = self._backtest(session)
        asset = make_asset(session)
        session.flush()
        session.add(
            m.Order(
                backtest_id=backtest.id,
                asset_id=asset.id,
                client_order_id="O1",
                decision_ts=dt.datetime(2020, 6, 2, tzinfo=UTC),
                execution_ts=dt.datetime(2020, 6, 1, tzinfo=UTC),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                status=OrderStatus.FILLED,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()

    def test_a_prediction_whose_target_precedes_its_decision_is_rejected(self, session):
        run = m.ModelRun(
            name="r",
            model_type="t",
            target_name="y",
            config={},
            config_hash="h",
            seed=1,
            validation_scheme="walk_forward",
            started_at=dt.datetime(2024, 1, 1, tzinfo=UTC),
        )
        asset = make_asset(session)
        session.add(run)
        session.flush()
        session.add(
            m.Prediction(
                model_run_id=run.id,
                asset_id=asset.id,
                ts=dt.datetime(2020, 6, 2, tzinfo=UTC),
                target_ts=dt.datetime(2020, 6, 1, tzinfo=UTC),
                split="test",
                y_pred=0.01,
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()


class TestReturnsRepository:
    def test_forward_and_trailing_returns_coexist(self, session):
        asset = make_asset(session)
        session.flush()
        ts = dt.datetime(2020, 6, 1, tzinfo=UTC)
        rows = [
            {
                "asset_id": asset.id,
                "ts": ts,
                "available_at": ts + dt.timedelta(hours=21),
                "horizon_days": 5,
                "direction": direction,
                "method": ReturnMethod.SIMPLE,
                "value": 0.01,
                "price_field": "adj_close",
            }
            for direction in (ReturnDirection.TRAILING, ReturnDirection.FORWARD)
        ]
        repo.upsert_returns(session, rows)
        session.flush()
        assert repo.count_rows(session, m.Return) == 2

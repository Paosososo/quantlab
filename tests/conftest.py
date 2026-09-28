"""Shared pytest fixtures.

Design notes
------------
*Unit tests run against SQLite in a temp file*, not PostgreSQL, so the suite is
fast and needs no services.  Foreign keys are switched on explicitly (SQLite
ignores them by default), so constraint tests are real.  Tests that genuinely
need PostgreSQL behaviour are marked ``integration`` and skipped unless a URL is
provided.

*No network, ever.*  Providers are exercised through the synthetic generator and
through checked-in fixture bytes.  A test suite that reaches the internet is a
test suite that goes red for reasons unrelated to the code.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from quantlab.config import get_settings, reset_settings
from quantlab.db.base import Base
from quantlab.db.session import build_engine, get_sessionmaker
from quantlab.ingestion.providers.synthetic import SyntheticSpec, generate_price_frame

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point every test at its own temp data directory and a throwaway database."""
    monkeypatch.setenv("QUANTLAB_ENV", "test")
    monkeypatch.setenv("QUANTLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("QUANTLAB_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("QUANTLAB_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("QUANTLAB_HTTP_BACKOFF_BASE_SECONDS", "0.001")
    monkeypatch.setenv("QUANTLAB_HTTP_BACKOFF_MAX_SECONDS", "0.01")
    reset_settings()
    get_settings().ensure_directories()
    yield
    reset_settings()


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    eng = build_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    factory = get_sessionmaker(engine)
    with factory() as sess:
        yield sess
        sess.rollback()


@pytest.fixture
def short_spec() -> SyntheticSpec:
    """Two years of daily bars: long enough for 252-day features, fast to build."""
    return SyntheticSpec(start=dt.date(2018, 1, 1), end=dt.date(2021, 12, 31))


@pytest.fixture
def price_frame(short_spec: SyntheticSpec) -> pd.DataFrame:
    return generate_price_frame("SYN_A", short_spec, seed=42)


@pytest.fixture
def two_symbol_frame(short_spec: SyntheticSpec) -> pd.DataFrame:
    return pd.concat(
        [
            generate_price_frame("SYN_A", short_spec, seed=42),
            generate_price_frame("SYN_B", short_spec, seed=43),
        ],
        ignore_index=True,
    )


@pytest.fixture
def postgres_url() -> str:
    url = os.environ.get("QUANTLAB_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("set QUANTLAB_TEST_POSTGRES_URL to run PostgreSQL integration tests")
    return url


def make_bars(
    symbol: str,
    closes: list[float],
    *,
    opens: list[float] | None = None,
    volumes: list[float] | None = None,
    start: dt.date = dt.date(2020, 1, 1),
) -> pd.DataFrame:
    """Hand-built bars for arithmetic tests.

    Deliberately simple and explicit: when a test asserts an exact number, the
    reader should be able to recompute it from the fixture by hand.
    """
    n = len(closes)
    dates = pd.bdate_range(start=start, periods=n, tz="UTC")
    opens = opens if opens is not None else list(closes)
    volumes = volumes if volumes is not None else [1_000_000.0] * n
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": dates,
            "available_at": dates + pd.Timedelta(hours=21),
            "open": opens,
            "high": [max(o, c) * 1.001 for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) * 0.999 for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            "adj_close": closes,
            "volume": volumes,
        }
    )

"""Engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from quantlab.config import get_settings
from quantlab.db.base import Base
from quantlab.logging import get_logger

log = get_logger(__name__)


def _configure_sqlite(engine: Engine) -> None:
    """Turn on the SQLite behaviours we rely on.

    SQLite does not enforce foreign keys unless asked, and the default
    journal mode serialises writers.  Unit tests run against SQLite, so the
    constraints they exercise must actually be enforced there or the tests are
    theatre.
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def build_engine(url: str | None = None, *, echo: bool | None = None) -> Engine:
    """Create a new engine.  Prefer :func:`get_engine` outside tests."""
    settings = get_settings()
    resolved_url = url or settings.sqlalchemy_url
    resolved_echo = settings.db_echo if echo is None else echo

    kwargs: dict[str, object] = {"echo": resolved_echo, "future": True, "pool_pre_ping": True}
    if resolved_url.startswith("postgresql"):
        kwargs["pool_size"] = settings.db_pool_size
        kwargs["max_overflow"] = settings.db_max_overflow

    engine = create_engine(resolved_url, **kwargs)
    if engine.dialect.name == "sqlite":
        _configure_sqlite(engine)
    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Process-wide engine singleton."""
    engine = build_engine()
    log.debug("engine.created", dialect=engine.dialect.name)
    return engine


def reset_engine() -> None:
    """Dispose and drop the cached engine.  Test-only helper.

    Written defensively because it runs during teardown, where raising turns one
    failing test into a confusing cascade.  ``get_engine`` may have been replaced
    by a fixture, in which case it has no cache to clear and there is nothing to
    do.
    """
    cache_info = getattr(get_engine, "cache_info", None)
    cache_clear = getattr(get_engine, "cache_clear", None)
    if cache_info is not None and cache_info().currsize:
        get_engine().dispose()
    if cache_clear is not None:
        cache_clear()


def get_sessionmaker(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Transactional scope.  Commits on success, rolls back on any exception.

    The exception is deliberately re-raised rather than logged and swallowed:
    a pipeline that silently continues after a failed write produces a database
    that disagrees with its own run log.
    """
    factory = get_sessionmaker(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all(engine: Engine | None = None) -> None:
    """Create every table.  Used by tests and by the SQLite quickstart path.

    Production schema changes go through Alembic; this function exists so unit
    tests do not need a migration run, and the two are kept consistent by a
    test that compares ``Base.metadata`` against the migrated schema.
    """
    Base.metadata.create_all(engine or get_engine())


def drop_all(engine: Engine | None = None) -> None:
    Base.metadata.drop_all(engine or get_engine())


def ping(engine: Engine | None = None) -> bool:
    """Return True when the database answers a trivial query."""
    try:
        with (engine or get_engine()).connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        log.warning("db.ping_failed", error=str(exc))
        return False
    return True


__all__ = [
    "build_engine",
    "create_all",
    "drop_all",
    "get_engine",
    "get_sessionmaker",
    "ping",
    "reset_engine",
    "session_scope",
]

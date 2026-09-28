"""Central configuration.

Every tunable lives here, is typed, and is overridable by an environment
variable prefixed ``QUANTLAB_``.  Nothing in the codebase reads ``os.environ``
directly; that keeps the surface area of "things that change behaviour" to a
single file, which is what makes a run reproducible from a recorded settings
snapshot.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from quantlab.exceptions import ConfigurationError

Environment = Literal["local", "ci", "docker", "test", "production"]


class Settings(BaseSettings):
    """Application settings, loaded from environment and/or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="QUANTLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- runtime ---------------------------------------------------------
    env: Environment = "local"
    log_level: str = "INFO"
    log_json: bool = False

    # --- database --------------------------------------------------------
    database_url: str | None = None
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "quantlab"
    postgres_password: str = "quantlab"
    postgres_db: str = "quantlab"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # --- storage ---------------------------------------------------------
    data_dir: Path = Path("./data")

    # --- HTTP / ingestion ------------------------------------------------
    http_timeout_seconds: float = 30.0
    http_max_retries: int = 5
    http_backoff_base_seconds: float = 0.5
    http_backoff_max_seconds: float = 30.0
    default_rate_limit_per_second: float = 4.0
    http_user_agent: str = "quantlab/0.1 (research; contact via repository)"

    # --- providers -------------------------------------------------------
    fred_api_key: str | None = None

    # --- research --------------------------------------------------------
    random_seed: int = 20240101
    trading_days_per_year: int = 252
    risk_free_rate_annual: float = 0.0

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        level = v.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ConfigurationError("unknown log level", value=v, allowed=sorted(allowed))
        return level

    @field_validator("trading_days_per_year")
    @classmethod
    def _positive_trading_days(cls, v: int) -> int:
        if v <= 0:
            raise ConfigurationError("trading_days_per_year must be positive", value=v)
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_url(self) -> str:
        """Resolved SQLAlchemy URL.

        ``database_url`` wins when set (that is how Docker and CI inject a URL);
        otherwise the discrete PostgreSQL parts are assembled.
        """
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    def ensure_directories(self) -> None:
        """Create the data layer directories if they do not exist."""
        for path in (self.raw_dir, self.processed_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, object]:
        """Settings snapshot safe to write into a run manifest or a log line."""
        secret_fields = {"postgres_password", "fred_api_key", "database_url"}
        out: dict[str, object] = {}
        for name in self.__class__.model_fields:
            value = getattr(self, name)
            if name in secret_fields and value:
                out[name] = "***redacted***"
            elif isinstance(value, Path):
                out[name] = str(value)
            else:
                out[name] = value
        return out


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that a run cannot silently pick up different settings halfway
    through.  Tests clear the cache via :func:`reset_settings`.
    """
    return Settings()


def reset_settings() -> None:
    """Drop the cached settings.  Test-only helper."""
    get_settings.cache_clear()


__all__ = ["Environment", "Settings", "get_settings", "reset_settings"]

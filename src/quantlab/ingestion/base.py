"""Provider abstractions.

The single most important design decision in this package is that **fetching
and parsing are separate operations**.

``fetch`` does I/O: it talks to the network, handles rate limits and retries,
and returns the provider's bytes untouched.  ``parse`` is a pure function from
those bytes to a canonical DataFrame.

Why split them?

* Parsing is where the subtle bugs live (column names, date formats, decimal
  separators, adjusted-vs-raw closes).  Keeping it pure means every parser is
  tested against a checked-in fixture with no network, deterministically, in
  milliseconds.
* The raw bytes are archived before parsing.  When a parser bug is found six
  months later, the fix is a reparse, not a re-download of data the provider may
  have since revised.
* Adding a provider means implementing two small methods, not subclassing a
  framework.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.db.enums import Frequency
from quantlab.timeutils import US_EQUITY_SESSION, SessionSpec

#: Canonical column set produced by every price parser.
PRICE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ts",
    "available_at",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

#: Canonical column set produced by every economic-series parser.
ECONOMIC_COLUMNS: tuple[str, ...] = ("code", "ts", "available_at", "value")


@dataclass(frozen=True, slots=True)
class RawPayload:
    """Bytes exactly as the provider returned them, plus provenance.

    ``sha256`` makes the raw layer content-addressed.  Re-running an ingestion
    that returns identical bytes writes to the same path, so the raw layer is
    idempotent by construction rather than by convention.
    """

    provider: str
    dataset: str
    entity: str
    content: bytes
    content_type: str
    fetched_at: dt.datetime
    request: dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def size_bytes(self) -> int:
        return len(self.content)

    def manifest(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "dataset": self.dataset,
            "entity": self.entity,
            "content_type": self.content_type,
            "fetched_at": self.fetched_at.isoformat(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "request": self.request,
        }


@dataclass(frozen=True, slots=True)
class SeriesMetadata:
    """What a provider knows about an economic series before any observations."""

    code: str
    title: str | None
    frequency: Frequency
    units: str | None
    publication_lag_days: int


class PriceProvider(ABC):
    """Daily OHLCV source."""

    #: Short, stable identifier stored in ``prices.source``.
    name: str = "abstract"
    dataset: str = "daily_bars"

    @abstractmethod
    async def fetch(
        self, symbol: str, start: dt.date | None = None, end: dt.date | None = None
    ) -> RawPayload:
        """Return the provider's raw response for one symbol."""

    @abstractmethod
    def parse(self, payload: RawPayload) -> pd.DataFrame:
        """Pure transformation from raw bytes to :data:`PRICE_COLUMNS`."""

    def session_for(self, symbol: str) -> SessionSpec:  # noqa: ARG002 - overridable hook
        """Which market session governs availability for ``symbol``."""
        return US_EQUITY_SESSION

    async def aclose(self) -> None:
        """Release network resources.  Default is a no-op."""


class EconomicProvider(ABC):
    """Macroeconomic time-series source."""

    name: str = "abstract"
    dataset: str = "economic_series"

    @abstractmethod
    async def fetch(
        self, code: str, start: dt.date | None = None, end: dt.date | None = None
    ) -> RawPayload: ...

    @abstractmethod
    def parse(self, payload: RawPayload, metadata: SeriesMetadata) -> pd.DataFrame:
        """Pure transformation from raw bytes to :data:`ECONOMIC_COLUMNS`."""

    @abstractmethod
    def metadata_for(self, code: str) -> SeriesMetadata:
        """Static metadata, including the conservative publication lag."""

    async def aclose(self) -> None: ...


__all__ = [
    "ECONOMIC_COLUMNS",
    "PRICE_COLUMNS",
    "EconomicProvider",
    "PriceProvider",
    "RawPayload",
    "SeriesMetadata",
]

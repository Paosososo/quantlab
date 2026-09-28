"""Stooq daily bars.

Why Stooq
---------
It is a genuinely free CSV endpoint with no API key, no registration and long
history for US equities, ETFs and major indices.  The alternatives considered:

* ``yfinance`` scrapes an undocumented endpoint that changes without notice and
  sits in a licensing grey area.  A portfolio project should not depend on it.
* Alpha Vantage, Tiingo, Polygon and friends are fine but require a key, which
  breaks the "clone and run" requirement.

Known limitations, documented rather than hidden
------------------------------------------------
* Stooq returns a single ``Close`` series per symbol.  It is adjusted for
  splits.  We do **not** assume it is adjusted for dividends, so this project
  sets ``adj_close = close`` and treats every return as a *price* return.  For
  dividend-paying assets that understates total return by roughly the dividend
  yield per year.  Every strategy and its benchmark are measured on the same
  basis, so relative comparisons stay valid; absolute levels do not.
* No delisted-symbol history is available, which caps how far survivorship bias
  can be addressed.  See ``docs/methodology.md``.
"""

from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from quantlab.exceptions import ProviderError
from quantlab.ingestion.base import PRICE_COLUMNS, PriceProvider, RawPayload
from quantlab.ingestion.http import AsyncHttpClient
from quantlab.logging import get_logger
from quantlab.timeutils import (
    LSE_SESSION,
    US_EQUITY_SESSION,
    UTC,
    SessionSpec,
    price_available_at,
    session_date_to_ts,
)

log = get_logger(__name__)

BASE_URL = "https://stooq.com/q/d/l/"

#: Suffix -> exchange session.  Stooq encodes the market in the ticker suffix.
_SESSION_BY_SUFFIX: dict[str, SessionSpec] = {
    "us": US_EQUITY_SESSION,
    "uk": LSE_SESSION,
}


class StooqProvider(PriceProvider):
    name = "stooq"
    dataset = "daily_bars"

    def __init__(self, client: AsyncHttpClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> AsyncHttpClient:
        if self._client is None:
            self._client = AsyncHttpClient(rate_limit_per_second=2.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def session_for(self, symbol: str) -> SessionSpec:
        suffix = symbol.rsplit(".", 1)[-1].lower() if "." in symbol else ""
        return _SESSION_BY_SUFFIX.get(suffix, US_EQUITY_SESSION)

    async def fetch(
        self, symbol: str, start: dt.date | None = None, end: dt.date | None = None
    ) -> RawPayload:
        params: dict[str, str] = {"s": symbol.lower(), "i": "d"}
        if start is not None:
            params["d1"] = start.strftime("%Y%m%d")
        if end is not None:
            params["d2"] = end.strftime("%Y%m%d")

        content = await self._get_client().get_bytes(BASE_URL, params=params)
        text = content.decode("utf-8", errors="replace").strip()
        # Stooq answers an unknown ticker with a 200 and a plain-text body.
        if not text or "No data" in text[:64] or not text.lower().startswith("date"):
            raise ProviderError("stooq returned no usable data", symbol=symbol, head=text[:80])
        return RawPayload(
            provider=self.name,
            dataset=self.dataset,
            entity=symbol,
            content=content,
            content_type="text/csv",
            fetched_at=dt.datetime.now(tz=UTC),
            request={"url": BASE_URL, "params": params},
        )

    def parse(self, payload: RawPayload) -> pd.DataFrame:
        """CSV -> canonical price frame.  Pure; tested against a stored fixture."""
        text = payload.content.decode("utf-8", errors="replace")
        try:
            raw = pd.read_csv(io.StringIO(text))
        except (pd.errors.ParserError, ValueError) as exc:
            raise ProviderError("could not parse stooq CSV", entity=payload.entity) from exc

        raw.columns = [c.strip().lower() for c in raw.columns]
        required = {"date", "open", "high", "low", "close"}
        if not required.issubset(raw.columns):
            raise ProviderError(
                "unexpected stooq columns",
                entity=payload.entity,
                columns=list(raw.columns),
            )

        session = self.session_for(payload.entity)
        dates = pd.to_datetime(raw["date"], errors="coerce").dt.date
        if dates.isna().any():
            raise ProviderError("unparseable dates in stooq CSV", entity=payload.entity)

        out = pd.DataFrame(
            {
                "symbol": payload.entity.upper(),
                "ts": [session_date_to_ts(d) for d in dates],
                "available_at": [price_available_at(d, session) for d in dates],
                "open": pd.to_numeric(raw["open"], errors="coerce"),
                "high": pd.to_numeric(raw["high"], errors="coerce"),
                "low": pd.to_numeric(raw["low"], errors="coerce"),
                "close": pd.to_numeric(raw["close"], errors="coerce"),
                "volume": pd.to_numeric(raw.get("volume"), errors="coerce")
                if "volume" in raw.columns
                else pd.Series([None] * len(raw), dtype="float64"),
            }
        )
        # See the module docstring: single close series, treated as price-only.
        out["adj_close"] = out["close"]
        out = out.dropna(subset=["open", "high", "low", "close"])
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        out["available_at"] = pd.to_datetime(out["available_at"], utc=True)
        return out.loc[:, list(PRICE_COLUMNS)].sort_values("ts").reset_index(drop=True)


__all__ = ["BASE_URL", "StooqProvider"]

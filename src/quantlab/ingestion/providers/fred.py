"""FRED macroeconomic series via the keyless CSV endpoint.

``https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`` needs no API
key, which keeps the project runnable straight after a clone.

The vintage problem, stated plainly
-----------------------------------
That endpoint returns the series **as it stands today**, not as it was first
published.  Two consequences:

1.  Revised values are what we see.  A model trained on revised GDP is using
    numbers nobody had at the time.
2.  There is no release date in the payload, so ``available_at`` has to be
    derived.

We handle (2) by attaching a conservative publication lag to each series and
setting ``available_at = observation_date + lag``.  The lags below are set at
the slow end of each series' typical release delay: being late is harmless,
being early is look-ahead bias.

We cannot fix (1) with this endpoint.  ALFRED (the vintage archive) exposes
real-time series and is the correct upgrade path; it is listed as future work
rather than silently ignored, and ``docs/methodology.md`` records that any
result involving revised macro data carries this caveat.
"""

from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from quantlab.db.enums import Frequency
from quantlab.exceptions import ProviderError
from quantlab.ingestion.base import (
    ECONOMIC_COLUMNS,
    EconomicProvider,
    RawPayload,
    SeriesMetadata,
)
from quantlab.ingestion.http import AsyncHttpClient
from quantlab.logging import get_logger
from quantlab.timeutils import UTC, macro_available_at, session_date_to_ts

log = get_logger(__name__)

BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

#: Conservative (deliberately long) publication lags in calendar days.
#: These are upper bounds on the usual delay, not exact release calendars.
KNOWN_SERIES: dict[str, SeriesMetadata] = {
    "DGS10": SeriesMetadata(
        "DGS10", "10-Year Treasury Constant Maturity Rate", Frequency.DAILY, "percent", 1
    ),
    "DGS2": SeriesMetadata(
        "DGS2", "2-Year Treasury Constant Maturity Rate", Frequency.DAILY, "percent", 1
    ),
    "DGS3MO": SeriesMetadata(
        "DGS3MO", "3-Month Treasury Constant Maturity Rate", Frequency.DAILY, "percent", 1
    ),
    "T10Y2Y": SeriesMetadata(
        "T10Y2Y", "10-Year minus 2-Year Treasury Spread", Frequency.DAILY, "percent", 1
    ),
    "T10Y3M": SeriesMetadata(
        "T10Y3M", "10-Year minus 3-Month Treasury Spread", Frequency.DAILY, "percent", 1
    ),
    "VIXCLS": SeriesMetadata("VIXCLS", "CBOE Volatility Index: VIX", Frequency.DAILY, "index", 1),
    "BAMLH0A0HYM2": SeriesMetadata(
        "BAMLH0A0HYM2",
        "ICE BofA US High Yield Index Option-Adjusted Spread",
        Frequency.DAILY,
        "percent",
        2,
    ),
    "DTWEXBGS": SeriesMetadata(
        "DTWEXBGS", "Nominal Broad U.S. Dollar Index", Frequency.DAILY, "index", 7
    ),
    "DFF": SeriesMetadata("DFF", "Federal Funds Effective Rate", Frequency.DAILY, "percent", 2),
    "UNRATE": SeriesMetadata("UNRATE", "Unemployment Rate", Frequency.MONTHLY, "percent", 45),
    "CPIAUCSL": SeriesMetadata(
        "CPIAUCSL", "CPI for All Urban Consumers: All Items", Frequency.MONTHLY, "index", 45
    ),
    "INDPRO": SeriesMetadata(
        "INDPRO", "Industrial Production: Total Index", Frequency.MONTHLY, "index", 45
    ),
    "PAYEMS": SeriesMetadata(
        "PAYEMS", "All Employees, Total Nonfarm", Frequency.MONTHLY, "thousands", 45
    ),
    "UMCSENT": SeriesMetadata(
        "UMCSENT", "University of Michigan: Consumer Sentiment", Frequency.MONTHLY, "index", 45
    ),
    "GDPC1": SeriesMetadata(
        "GDPC1", "Real Gross Domestic Product", Frequency.QUARTERLY, "billions_chained_2017", 120
    ),
}

DEFAULT_LAG_DAYS = 45


class FredProvider(EconomicProvider):
    name = "fred"
    dataset = "economic_series"

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

    def metadata_for(self, code: str) -> SeriesMetadata:
        known = KNOWN_SERIES.get(code.upper())
        if known is not None:
            return known
        # Unknown series still work, with the conservative default lag and a
        # loud log line so it is obvious the lag was guessed rather than known.
        log.warning("fred.unknown_series_lag_default", code=code, lag_days=DEFAULT_LAG_DAYS)
        return SeriesMetadata(code.upper(), None, Frequency.MONTHLY, None, DEFAULT_LAG_DAYS)

    async def fetch(
        self, code: str, start: dt.date | None = None, end: dt.date | None = None
    ) -> RawPayload:
        params: dict[str, str] = {"id": code.upper()}
        if start is not None:
            params["cosd"] = start.isoformat()
        if end is not None:
            params["coed"] = end.isoformat()

        content = await self._get_client().get_bytes(BASE_URL, params=params)
        head = content[:200].decode("utf-8", errors="replace").lower()
        if "," not in head:
            raise ProviderError("fred returned a non-CSV body", code=code, head=head[:80])
        return RawPayload(
            provider=self.name,
            dataset=self.dataset,
            entity=code.upper(),
            content=content,
            content_type="text/csv",
            fetched_at=dt.datetime.now(tz=UTC),
            request={"url": BASE_URL, "params": params},
        )

    def parse(self, payload: RawPayload, metadata: SeriesMetadata) -> pd.DataFrame:
        text = payload.content.decode("utf-8", errors="replace")
        try:
            raw = pd.read_csv(io.StringIO(text))
        except (pd.errors.ParserError, ValueError) as exc:
            raise ProviderError("could not parse FRED CSV", code=payload.entity) from exc

        if raw.shape[1] < 2:
            raise ProviderError("FRED CSV has too few columns", code=payload.entity)

        # The header has changed over time: older exports use DATE, newer ones
        # observation_date.  Take the first column positionally instead of
        # guessing a name, and the value column by position too.
        date_col, value_col = raw.columns[0], raw.columns[1]
        dates = pd.to_datetime(raw[date_col], errors="coerce").dt.date
        if dates.isna().all():
            raise ProviderError("no parseable dates in FRED CSV", code=payload.entity)

        # FRED writes '.' for a missing observation.
        values = pd.to_numeric(raw[value_col].replace(".", pd.NA), errors="coerce")

        out = pd.DataFrame(
            {
                "code": metadata.code,
                "ts": [session_date_to_ts(d) if pd.notna(d) else pd.NaT for d in dates],
                "available_at": [
                    macro_available_at(d, metadata.publication_lag_days) if pd.notna(d) else pd.NaT
                    for d in dates
                ],
                "value": values.astype("float64"),
            }
        )
        out = out.dropna(subset=["ts"])
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        out["available_at"] = pd.to_datetime(out["available_at"], utc=True)
        return out.loc[:, list(ECONOMIC_COLUMNS)].sort_values("ts").reset_index(drop=True)


__all__ = ["BASE_URL", "DEFAULT_LAG_DAYS", "KNOWN_SERIES", "FredProvider"]

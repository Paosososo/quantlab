"""Ingestion orchestration: fetch, archive, validate, load.

Concurrency model
-----------------
Network fetches run concurrently under a semaphore; database writes run
sequentially in the calling thread.  Reaching for an async database driver here
would buy nothing: the bottleneck is provider latency and rate limits, not
INSERT throughput, and a synchronous write path keeps transaction boundaries
obvious.  Boring where it can be, concurrent where it pays.

Idempotency
-----------
Three mechanisms, layered:

1.  The raw layer is content-addressed, so re-fetching identical bytes is a
    no-op on disk.
2.  Loads are ``INSERT ... ON CONFLICT DO UPDATE`` on the natural key, so
    re-running a window overwrites rather than duplicating.
3.  Incremental runs start from the database watermark minus an overlap window,
    not from the scheduler's execution date.  A backfill, a retry and a normal
    run therefore all converge to the same database state.

The overlap window matters: providers revise recent bars.  Re-reading the last
few sessions each run picks up corrections that a strict "everything after the
watermark" rule would miss forever.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from quantlab.db import models as m
from quantlab.db import repository as repo
from quantlab.db.enums import AssetClass, ReturnDirection, ReturnMethod, RunStatus
from quantlab.exceptions import IngestionError
from quantlab.ingestion.base import (
    ECONOMIC_COLUMNS,
    PRICE_COLUMNS,
    EconomicProvider,
    PriceProvider,
    RawPayload,
)
from quantlab.ingestion.raw_store import RawStore
from quantlab.ingestion.validation import (
    ValidationReport,
    deduplicate,
    run_economic_quality_checks,
    run_price_quality_checks,
    validate_schema,
)
from quantlab.logging import get_logger
from quantlab.timeutils import UTC, to_utc

log = get_logger(__name__)

#: Re-read this many calendar days before the watermark to absorb provider
#: revisions to recently published bars.
DEFAULT_OVERLAP_DAYS = 5


@dataclass(slots=True)
class SymbolResult:
    symbol: str
    status: RunStatus
    rows_fetched: int = 0
    rows_written: int = 0
    rows_rejected: int = 0
    raw_path: str | None = None
    payload_sha256: str | None = None
    report: ValidationReport | None = None
    error: str | None = None


@dataclass(slots=True)
class IngestionSummary:
    provider: str
    dataset: str
    results: list[SymbolResult] = field(default_factory=list)

    @property
    def succeeded(self) -> list[SymbolResult]:
        return [r for r in self.results if r.status is RunStatus.SUCCEEDED]

    @property
    def failed(self) -> list[SymbolResult]:
        return [r for r in self.results if r.status is RunStatus.FAILED]

    @property
    def rows_written(self) -> int:
        return sum(r.rows_written for r in self.results)

    def raise_if_all_failed(self) -> None:
        """Fail the task when nothing at all loaded.

        Partial failure is tolerated (one bad ticker should not sink a daily
        run) but total failure is escalated, because an Airflow task that goes
        green after writing zero rows is worse than no task at all.
        """
        if self.results and not self.succeeded:
            raise IngestionError(
                "every entity failed",
                provider=self.provider,
                dataset=self.dataset,
                errors=[r.error for r in self.failed][:5],
            )


# ---------------------------------------------------------------------------
# Async fetch fan-out
# ---------------------------------------------------------------------------
async def _fetch_many(
    provider: PriceProvider | EconomicProvider,
    entities: Sequence[str],
    windows: dict[str, tuple[dt.date | None, dt.date | None]],
    *,
    max_concurrency: int = 4,
) -> dict[str, RawPayload | BaseException]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(entity: str) -> RawPayload:
        start, end = windows.get(entity, (None, None))
        async with semaphore:
            return await provider.fetch(entity, start, end)

    tasks = [one(e) for e in entities]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip(entities, gathered, strict=True))


def _run_fetches(
    provider: PriceProvider | EconomicProvider,
    entities: Sequence[str],
    windows: dict[str, tuple[dt.date | None, dt.date | None]],
    *,
    max_concurrency: int = 4,
) -> dict[str, RawPayload | BaseException]:
    """Bridge from the synchronous pipeline into the async fetch layer."""

    async def runner() -> dict[str, RawPayload | BaseException]:
        try:
            return await _fetch_many(provider, entities, windows, max_concurrency=max_concurrency)
        finally:
            await provider.aclose()

    return asyncio.run(runner())


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def _price_window(
    session: Session, symbol: str, start: dt.date | None, end: dt.date | None, overlap_days: int
) -> tuple[dt.date | None, dt.date | None]:
    if start is not None:
        return start, end
    asset = repo.get_asset_by_symbol(session, symbol)
    if asset is None:
        return None, end
    watermark = repo.latest_price_ts(session, asset.id)
    if watermark is None:
        return None, end
    resume = (to_utc(watermark) - dt.timedelta(days=overlap_days)).date()
    return resume, end


def ingest_prices(
    session: Session,
    provider: PriceProvider,
    symbols: Sequence[str],
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
    asset_class: AssetClass = AssetClass.EQUITY,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    raw_store: RawStore | None = None,
    max_concurrency: int = 4,
    persist_quality: bool = True,
) -> IngestionSummary:
    """Fetch, archive, validate and load daily bars for ``symbols``."""
    store = raw_store or RawStore()
    summary = IngestionSummary(provider=provider.name, dataset=provider.dataset)
    windows = {s: _price_window(session, s, start, end, overlap_days) for s in symbols}

    started_at = dt.datetime.now(tz=UTC)
    payloads = _run_fetches(provider, list(symbols), windows, max_concurrency=max_concurrency)

    for symbol in symbols:
        run = m.IngestionRun(
            provider=provider.name,
            dataset=provider.dataset,
            entity=symbol,
            started_at=started_at,
            status=RunStatus.RUNNING,
        )
        session.add(run)
        session.flush()

        result = SymbolResult(symbol=symbol, status=RunStatus.FAILED)
        payload = payloads.get(symbol)
        try:
            if isinstance(payload, BaseException):
                raise payload
            if payload is None:
                raise IngestionError("no payload returned", symbol=symbol)

            raw_path = store.write(payload)
            result.raw_path = str(raw_path)
            result.payload_sha256 = payload.sha256

            frame = provider.parse(payload)
            frame = validate_schema(
                frame,
                required=PRICE_COLUMNS,
                numeric=("open", "high", "low", "close", "adj_close", "volume"),
                datetime_utc=("ts", "available_at"),
                label=f"prices:{symbol}",
            )
            result.rows_fetched = len(frame)
            frame, dropped = deduplicate(frame, ["symbol", "ts"], label=f"prices:{symbol}")
            result.rows_rejected += dropped
            frame = frame.sort_values("ts").reset_index(drop=True)

            report = run_price_quality_checks(frame, entity=symbol)
            result.report = report
            if persist_quality:
                repo.bulk_insert(session, m.DataQualityCheck, report.to_rows(run.id))
            if not report.ok:
                raise IngestionError(
                    "data quality gate failed",
                    symbol=symbol,
                    failures=[c.name for c in report.failures],
                )

            asset = repo.get_or_create_asset(
                session,
                symbol.upper(),
                asset_class=asset_class,
                session_name=provider.session_for(symbol).name,
            )
            rows = _price_rows(frame, asset_id=asset.id, source=provider.name)
            result.rows_written = repo.upsert_prices(session, rows)

            run.status = RunStatus.SUCCEEDED
            result.status = RunStatus.SUCCEEDED
            if len(frame):
                run.watermark_from = frame["ts"].iloc[0].to_pydatetime()
                run.watermark_to = frame["ts"].iloc[-1].to_pydatetime()
            log.info(
                "ingest.prices.ok",
                symbol=symbol,
                rows=result.rows_written,
                warnings=[c.name for c in report.warnings],
            )
        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            result.error = run.error
            log.error("ingest.prices.failed", symbol=symbol, error=run.error)
        finally:
            run.finished_at = dt.datetime.now(tz=UTC)
            run.rows_fetched = result.rows_fetched
            run.rows_written = result.rows_written
            run.rows_rejected = result.rows_rejected
            run.raw_path = result.raw_path
            run.payload_sha256 = result.payload_sha256
            session.flush()
            summary.results.append(result)

    return summary


def _price_rows(frame: pd.DataFrame, *, asset_id: int, source: str) -> list[dict[str, Any]]:
    now = dt.datetime.now(tz=UTC)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        rows.append(
            {
                "asset_id": asset_id,
                "ts": record["ts"].to_pydatetime(),
                "available_at": record["available_at"].to_pydatetime(),
                "open": repo.to_decimal(record["open"]),
                "high": repo.to_decimal(record["high"]),
                "low": repo.to_decimal(record["low"]),
                "close": repo.to_decimal(record["close"]),
                "adj_close": repo.to_decimal(record.get("adj_close")),
                "volume": repo.to_decimal(record.get("volume"), places=10),
                "source": source,
                "ingested_at": now,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Economic data
# ---------------------------------------------------------------------------
def ingest_economic(
    session: Session,
    provider: EconomicProvider,
    codes: Sequence[str],
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
    raw_store: RawStore | None = None,
    max_concurrency: int = 3,
    persist_quality: bool = True,
) -> IngestionSummary:
    store = raw_store or RawStore()
    summary = IngestionSummary(provider=provider.name, dataset=provider.dataset)
    windows = dict.fromkeys(codes, (start, end))
    started_at = dt.datetime.now(tz=UTC)
    payloads = _run_fetches(provider, list(codes), windows, max_concurrency=max_concurrency)

    for code in codes:
        run = m.IngestionRun(
            provider=provider.name,
            dataset=provider.dataset,
            entity=code,
            started_at=started_at,
            status=RunStatus.RUNNING,
        )
        session.add(run)
        session.flush()
        result = SymbolResult(symbol=code, status=RunStatus.FAILED)
        payload = payloads.get(code)
        try:
            if isinstance(payload, BaseException):
                raise payload
            if payload is None:
                raise IngestionError("no payload returned", code=code)

            raw_path = store.write(payload)
            result.raw_path = str(raw_path)
            result.payload_sha256 = payload.sha256

            metadata = provider.metadata_for(code)
            frame = provider.parse(payload, metadata)
            frame = validate_schema(
                frame,
                required=ECONOMIC_COLUMNS,
                numeric=("value",),
                datetime_utc=("ts", "available_at"),
                label=f"economic:{code}",
            )
            result.rows_fetched = len(frame)
            frame, dropped = deduplicate(frame, ["code", "ts"], label=f"economic:{code}")
            result.rows_rejected += dropped

            report = run_economic_quality_checks(frame, entity=code)
            result.report = report
            if persist_quality:
                repo.bulk_insert(session, m.DataQualityCheck, report.to_rows(run.id))
            if not report.ok:
                raise IngestionError(
                    "data quality gate failed",
                    code=code,
                    failures=[c.name for c in report.failures],
                )

            series = repo.get_or_create_series(
                session,
                metadata.code,
                frequency=metadata.frequency,
                title=metadata.title,
                units=metadata.units,
                publication_lag_days=metadata.publication_lag_days,
                source=provider.name,
            )
            now = dt.datetime.now(tz=UTC)
            rows = [
                {
                    "series_id": series.id,
                    "ts": r["ts"].to_pydatetime(),
                    "available_at": r["available_at"].to_pydatetime(),
                    "value": None if pd.isna(r["value"]) else float(r["value"]),
                    "source": provider.name,
                    "ingested_at": now,
                }
                for r in frame.to_dict("records")
            ]
            result.rows_written = repo.upsert_economic(session, rows)
            run.status = RunStatus.SUCCEEDED
            result.status = RunStatus.SUCCEEDED
            if len(frame):
                run.watermark_from = frame["ts"].iloc[0].to_pydatetime()
                run.watermark_to = frame["ts"].iloc[-1].to_pydatetime()
            log.info("ingest.economic.ok", code=code, rows=result.rows_written)
        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            result.error = run.error
            log.error("ingest.economic.failed", code=code, error=run.error)
        finally:
            run.finished_at = dt.datetime.now(tz=UTC)
            run.rows_fetched = result.rows_fetched
            run.rows_written = result.rows_written
            run.rows_rejected = result.rows_rejected
            run.raw_path = result.raw_path
            run.payload_sha256 = result.payload_sha256
            session.flush()
            summary.results.append(result)

    return summary


# ---------------------------------------------------------------------------
# Derived returns
# ---------------------------------------------------------------------------
def materialise_returns(
    session: Session,
    symbols: Sequence[str],
    *,
    horizons: Sequence[int] = (1, 5, 21),
    method: ReturnMethod = ReturnMethod.SIMPLE,
    price_field: str = "adj_close",
    include_forward: bool = True,
) -> int:
    """Compute and store trailing and forward returns.

    Trailing return at ``t`` uses prices at ``t-h`` and ``t``; it is knowable at
    ``t``'s availability.  Forward return at ``t`` uses ``t`` and ``t+h``; it is
    knowable only at ``t+h``'s availability, and that is exactly what its
    ``available_at`` records.  Storing both in one table with an explicit
    ``direction`` makes the distinction impossible to lose in a join.
    """
    written = 0
    for symbol in symbols:
        asset = repo.get_asset_by_symbol(session, symbol.upper())
        if asset is None:
            log.warning("returns.unknown_symbol", symbol=symbol)
            continue
        prices = repo.load_prices(session, [symbol.upper()])
        if prices.empty:
            continue
        prices = prices.sort_values("ts").reset_index(drop=True)
        px = prices[price_field].to_numpy(dtype="float64")
        ts = prices["ts"]
        avail = prices["available_at"]

        rows: list[dict[str, Any]] = []
        for h in horizons:
            if len(px) <= h:
                continue
            if method is ReturnMethod.SIMPLE:
                # ``fill_method=None``: a gap must stay NaN.  Pandas' legacy
                # default forward-filled prices first, which reports a 0%
                # return across a hole in the data instead of "unknown".
                trailing = pd.Series(px).pct_change(h, fill_method=None)
            else:
                trailing = pd.Series(px).transform("log").diff(h)

            for i in range(h, len(px)):
                value = float(trailing.iloc[i])
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "asset_id": asset.id,
                        "ts": ts.iloc[i].to_pydatetime(),
                        "available_at": avail.iloc[i].to_pydatetime(),
                        "horizon_days": h,
                        "direction": ReturnDirection.TRAILING,
                        "method": method,
                        "value": value,
                        "price_field": price_field,
                    }
                )

            if include_forward:
                for i in range(0, len(px) - h):
                    if method is ReturnMethod.SIMPLE:
                        value = float(px[i + h] / px[i] - 1.0)
                    else:
                        value = float(pd.Series([px[i], px[i + h]]).transform("log").diff().iloc[1])
                    rows.append(
                        {
                            "asset_id": asset.id,
                            "ts": ts.iloc[i].to_pydatetime(),
                            # Knowable only once the far end of the window is public.
                            "available_at": avail.iloc[i + h].to_pydatetime(),
                            "horizon_days": h,
                            "direction": ReturnDirection.FORWARD,
                            "method": method,
                            "value": value,
                            "price_field": price_field,
                        }
                    )
        written += repo.upsert_returns(session, rows)
        log.info("returns.materialised", symbol=symbol, rows=len(rows))
    return written


__all__ = [
    "DEFAULT_OVERLAP_DAYS",
    "IngestionSummary",
    "SymbolResult",
    "ingest_economic",
    "ingest_prices",
    "materialise_returns",
]

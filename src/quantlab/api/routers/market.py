"""Assets, prices, returns and macroeconomic series."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Path, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from quantlab.api.deps import AsOfDep, Pagination, PaginationDep, SessionDep
from quantlab.api.errors import NotFoundError
from quantlab.api.schemas import (
    AssetDetail,
    AssetOut,
    EconomicObservationOut,
    Page,
    PriceOut,
    ReturnOut,
)
from quantlab.db import models as m
from quantlab.db import repository as repo
from quantlab.db.enums import ReturnDirection, ReturnMethod

router = APIRouter(tags=["market data"])

SYMBOL = Path(description="Ticker symbol, case-insensitive")


@router.get("/assets", response_model=Page[AssetOut])
def list_assets(
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
    asset_class: str | None = Query(default=None),
    search: str | None = Query(default=None, description="case-insensitive substring match"),
) -> Page[AssetOut]:
    stmt = select(m.Asset).order_by(m.Asset.symbol)
    if asset_class:
        stmt = stmt.where(m.Asset.asset_class == asset_class)
    if search:
        stmt = stmt.where(m.Asset.symbol.ilike(f"%{search}%"))
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all()
    items = [AssetOut.model_validate(r) for r in rows]
    return Page[AssetOut](
        items=items, limit=page.limit, offset=page.offset, returned=len(items), total=int(total)
    )


@router.get("/assets/{symbol}", response_model=AssetDetail)
def get_asset(symbol: str = SYMBOL, session: Session = SessionDep) -> AssetDetail:
    asset = repo.get_asset_by_symbol(session, symbol.upper())
    if asset is None:
        raise NotFoundError("asset not found", symbol=symbol)
    first, last, count = repo.price_coverage(session, asset.symbol)
    detail = AssetDetail.model_validate(asset)
    detail.first_price_ts = first
    detail.last_price_ts = last
    detail.n_prices = count
    return detail


@router.get("/assets/{symbol}/prices", response_model=Page[PriceOut])
def get_prices(
    symbol: str = SYMBOL,
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
    start: dt.datetime | None = Query(default=None),
    end: dt.datetime | None = Query(default=None),
    as_of: dt.datetime | None = AsOfDep,
) -> Page[PriceOut]:
    """Daily bars.

    Supplying ``as_of`` applies the availability filter, so the response is the
    history as it stood at that instant.  Omitting it returns everything, which
    is right for reporting and wrong for building a feature.
    """
    asset = repo.get_asset_by_symbol(session, symbol.upper())
    if asset is None:
        raise NotFoundError("asset not found", symbol=symbol)

    frame = repo.load_prices(session, [asset.symbol], start=start, end=end, as_of=as_of)
    window = frame.iloc[page.offset : page.offset + page.limit]
    items = [
        PriceOut(
            symbol=row["symbol"],
            ts=row["ts"].to_pydatetime(),
            available_at=row["available_at"].to_pydatetime(),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            adj_close=row.get("adj_close"),
            volume=row.get("volume"),
        )
        for row in window.to_dict("records")
    ]
    return Page[PriceOut](
        items=items,
        limit=page.limit,
        offset=page.offset,
        returned=len(items),
        total=int(len(frame)),
    )


@router.get("/assets/{symbol}/returns", response_model=Page[ReturnOut])
def get_returns(
    symbol: str = SYMBOL,
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
    horizon_days: int = Query(default=1, ge=1, le=365),
    direction: ReturnDirection = Query(default=ReturnDirection.TRAILING),
    method: ReturnMethod = Query(default=ReturnMethod.SIMPLE),
    as_of: dt.datetime | None = AsOfDep,
) -> Page[ReturnOut]:
    """Stored returns.

    ``direction=forward`` returns labels, whose ``available_at`` is the end of
    the forward window.  Filtering those with ``as_of`` is how a caller confirms
    a label was actually knowable at a given decision time.
    """
    asset = repo.get_asset_by_symbol(session, symbol.upper())
    if asset is None:
        raise NotFoundError("asset not found", symbol=symbol)

    frame = repo.load_returns(
        session,
        [asset.symbol],
        horizon_days=horizon_days,
        direction=direction,
        method=method,
        as_of=as_of,
    )
    window = frame.iloc[page.offset : page.offset + page.limit]
    items = [
        ReturnOut(
            symbol=row["symbol"],
            ts=row["ts"].to_pydatetime(),
            available_at=row["available_at"].to_pydatetime(),
            horizon_days=horizon_days,
            direction=str(direction),
            method=str(method),
            value=float(row["value"]),
        )
        for row in window.to_dict("records")
    ]
    return Page[ReturnOut](
        items=items,
        limit=page.limit,
        offset=page.offset,
        returned=len(items),
        total=int(len(frame)),
    )


@router.get("/economic/{code}", response_model=Page[EconomicObservationOut])
def get_economic_series(
    code: str = Path(description="FRED series code, e.g. DGS10"),
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
    as_of: dt.datetime | None = AsOfDep,
) -> Page[EconomicObservationOut]:
    series = session.execute(
        select(m.EconomicSeries).where(m.EconomicSeries.code == code.upper())
    ).scalar_one_or_none()
    if series is None:
        raise NotFoundError("economic series not found", code=code)

    frame = repo.load_economic(session, [series.code], as_of=as_of)
    window = frame.iloc[page.offset : page.offset + page.limit]
    items = [
        EconomicObservationOut(
            code=row["code"],
            ts=row["ts"].to_pydatetime(),
            available_at=row["available_at"].to_pydatetime(),
            value=None if row["value"] is None else float(row["value"]),
        )
        for row in window.to_dict("records")
    ]
    return Page[EconomicObservationOut](
        items=items,
        limit=page.limit,
        offset=page.offset,
        returned=len(items),
        total=int(len(frame)),
    )


__all__ = ["router"]

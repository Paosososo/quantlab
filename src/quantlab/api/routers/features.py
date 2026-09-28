"""Feature sets and feature values."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Path, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from quantlab.api.deps import AsOfDep, Pagination, PaginationDep, SessionDep
from quantlab.api.errors import NotFoundError
from quantlab.api.schemas import FeatureSetOut, FeatureValueOut, Page
from quantlab.db import models as m
from quantlab.db import repository as repo

router = APIRouter(prefix="/feature-sets", tags=["features"])


@router.get("", response_model=list[FeatureSetOut])
def list_feature_sets(session: Session = SessionDep) -> list[FeatureSetOut]:
    rows = (
        session.execute(select(m.FeatureSet).order_by(m.FeatureSet.name, m.FeatureSet.version))
        .scalars()
        .all()
    )
    return [
        FeatureSetOut(
            id=row.id,
            name=row.name,
            version=row.version,
            spec_hash=row.spec_hash,
            description=row.description,
            n_features=len(row.definition.get("features", [])),
        )
        for row in rows
    ]


@router.get("/{feature_set_id}", response_model=dict)
def get_feature_set(feature_set_id: int = Path(ge=1), session: Session = SessionDep) -> dict:
    """Full definition, including every feature's parameters and lookback.

    Returned verbatim so a reader can reproduce the feature set from the API
    alone; the spec hash is a hash of exactly this document.
    """
    row = session.get(m.FeatureSet, feature_set_id)
    if row is None:
        raise NotFoundError("feature set not found", feature_set_id=feature_set_id)
    return {
        "id": row.id,
        "name": row.name,
        "version": row.version,
        "spec_hash": row.spec_hash,
        "description": row.description,
        "definition": row.definition,
    }


@router.get("/{feature_set_id}/values", response_model=Page[FeatureValueOut])
def get_feature_values(
    feature_set_id: int = Path(ge=1),
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
    symbol: str | None = Query(default=None),
    names: list[str] | None = Query(default=None, description="feature names to return"),
    as_of: dt.datetime | None = AsOfDep,
) -> Page[FeatureValueOut]:
    if session.get(m.FeatureSet, feature_set_id) is None:
        raise NotFoundError("feature set not found", feature_set_id=feature_set_id)

    frame = repo.load_features(
        session,
        feature_set_id,
        [symbol.upper()] if symbol else None,
        names=names,
        as_of=as_of,
    )
    window = frame.iloc[page.offset : page.offset + page.limit]
    items = [
        FeatureValueOut(
            symbol=row["symbol"],
            name=row["name"],
            ts=row["ts"].to_pydatetime(),
            available_at=row["available_at"].to_pydatetime(),
            value=None if row["value"] is None else float(row["value"]),
        )
        for row in window.to_dict("records")
    ]
    return Page[FeatureValueOut](
        items=items,
        limit=page.limit,
        offset=page.offset,
        returned=len(items),
        total=int(len(frame)),
    )


__all__ = ["router"]

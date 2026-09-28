"""Model runs, predictions and the comparison table."""

from __future__ import annotations

from fastapi import APIRouter, Path, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from quantlab.api.deps import Pagination, PaginationDep, SessionDep
from quantlab.api.errors import NotFoundError
from quantlab.api.schemas import ModelComparisonRow, ModelRunOut, Page, PredictionOut
from quantlab.db import models as m

router = APIRouter(prefix="/model-runs", tags=["models"])


@router.get("", response_model=Page[ModelRunOut])
def list_model_runs(
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
    model_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
) -> Page[ModelRunOut]:
    stmt = select(m.ModelRun).order_by(m.ModelRun.started_at.desc())
    if model_type:
        stmt = stmt.where(m.ModelRun.model_type == model_type)
    if status:
        stmt = stmt.where(m.ModelRun.status == status)
    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.limit(page.limit).offset(page.offset)).scalars().all()
    items = [ModelRunOut.model_validate(r) for r in rows]
    return Page[ModelRunOut](
        items=items, limit=page.limit, offset=page.offset, returned=len(items), total=int(total)
    )


@router.get("/comparison", response_model=list[ModelComparisonRow])
def comparison(
    session: Session = SessionDep,
    limit: int = Query(default=20, ge=1, le=200),
) -> list[ModelComparisonRow]:
    """The headline table for the research question.

    One row per model run with the metrics that answer "did this beat the
    baseline?".  Ordered by out-of-sample R-squared, which is the comparison
    that matters; the ordering is not a claim that the top row is good.
    """
    rows = (
        session.execute(
            select(m.ModelRun)
            .where(m.ModelRun.metrics.is_not(None))
            .order_by(m.ModelRun.started_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    out = []
    for row in rows:
        metrics = row.metrics or {}
        out.append(
            ModelComparisonRow(
                model_run_id=row.id,
                model_type=row.model_type,
                rmse=metrics.get("rmse"),
                r2_oos_vs_zero=metrics.get("r2_oos_vs_zero"),
                information_coefficient=metrics.get("information_coefficient"),
                directional_accuracy=metrics.get("directional_accuracy"),
                n=metrics.get("n"),
            )
        )
    return sorted(
        out,
        key=lambda r: r.r2_oos_vs_zero if r.r2_oos_vs_zero is not None else -1e9,
        reverse=True,
    )


@router.get("/{model_run_id}", response_model=ModelRunOut)
def get_model_run(model_run_id: int = Path(ge=1), session: Session = SessionDep) -> ModelRunOut:
    row = session.get(m.ModelRun, model_run_id)
    if row is None:
        raise NotFoundError("model run not found", model_run_id=model_run_id)
    return ModelRunOut.model_validate(row)


@router.get("/{model_run_id}/predictions", response_model=Page[PredictionOut])
def get_predictions(
    model_run_id: int = Path(ge=1),
    session: Session = SessionDep,
    page: Pagination = PaginationDep,
    symbol: str | None = Query(default=None),
    fold: int | None = Query(default=None, ge=0),
) -> Page[PredictionOut]:
    if session.get(m.ModelRun, model_run_id) is None:
        raise NotFoundError("model run not found", model_run_id=model_run_id)

    stmt = (
        select(
            m.Asset.symbol,
            m.Prediction.ts,
            m.Prediction.target_ts,
            m.Prediction.fold,
            m.Prediction.split,
            m.Prediction.y_pred,
            m.Prediction.y_true,
            m.Prediction.y_proba,
        )
        .join(m.Asset, m.Asset.id == m.Prediction.asset_id)
        .where(m.Prediction.model_run_id == model_run_id)
        .order_by(m.Prediction.ts, m.Asset.symbol)
    )
    if symbol:
        stmt = stmt.where(m.Asset.symbol == symbol.upper())
    if fold is not None:
        stmt = stmt.where(m.Prediction.fold == fold)

    total = session.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
    rows = session.execute(stmt.limit(page.limit).offset(page.offset)).mappings().all()
    items = [
        PredictionOut(
            symbol=r["symbol"],
            ts=r["ts"],
            target_ts=r["target_ts"],
            fold=r["fold"],
            split=str(r["split"]),
            y_pred=float(r["y_pred"]),
            y_true=None if r["y_true"] is None else float(r["y_true"]),
            y_proba=None if r["y_proba"] is None else float(r["y_proba"]),
        )
        for r in rows
    ]
    return Page[PredictionOut](
        items=items, limit=page.limit, offset=page.offset, returned=len(items), total=int(total)
    )


__all__ = ["router"]

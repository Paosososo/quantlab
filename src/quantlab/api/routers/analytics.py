"""Portfolio analytics computed on demand.

Two POST endpoints that do real work rather than reading a table.  Both accept
an ``as_of`` cutoff, which is the point: an optimiser or a risk model queried
without one is silently using data from after the date it claims to describe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter
from sqlalchemy.orm import Session

from quantlab.api.deps import SessionDep
from quantlab.api.errors import NotFoundError
from quantlab.api.schemas import OptimisationRequest, OptimisationResponse, RiskRequest
from quantlab.db import repository as repo
from quantlab.exceptions import InsufficientDataError
from quantlab.risk.metrics import diversification_ratio, shrunk_covariance, summarise_risk
from quantlab.risk.optimization import OPTIMISERS, risk_contributions

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.post("/optimise", response_model=OptimisationResponse)
def optimise_portfolio(
    request: OptimisationRequest, session: Session = SessionDep
) -> OptimisationResponse:
    """Compute portfolio weights from stored prices.

    ``minimum_variance`` is the default rather than maximum-Sharpe, because
    maximum-Sharpe needs expected returns and those estimates are dominated by
    noise; see :mod:`quantlab.risk.optimization` for the full argument.
    """
    optimiser = OPTIMISERS.get(request.method)
    if optimiser is None:
        raise InsufficientDataError(
            "unknown optimisation method", method=request.method, known=sorted(OPTIMISERS)
        )

    symbols = [s.upper() for s in request.symbols]
    prices = repo.load_prices(session, symbols, as_of=request.as_of)
    if prices.empty:
        raise NotFoundError("no price data for the requested symbols", symbols=symbols)

    wide = repo.wide_close_frame(prices).tail(request.lookback_days + 1)
    returns = wide.pct_change(fill_method=None).dropna()
    missing = [s for s in symbols if s not in returns.columns]
    if missing:
        raise NotFoundError("some symbols have no usable history", missing=missing)
    if len(returns) < len(symbols) + 20:
        raise InsufficientDataError(
            "not enough overlapping observations to estimate a covariance",
            observations=int(len(returns)),
            assets=len(symbols),
        )

    covariance = shrunk_covariance(returns, shrinkage=request.shrinkage)
    expected = returns.mean() * 252
    result = optimiser(covariance, expected_returns=expected)
    contributions = risk_contributions(result.weights, covariance)

    return OptimisationResponse(
        method=result.method,
        as_of=request.as_of,
        n_observations=int(len(returns)),
        weights={k: float(v) for k, v in result.weights.items()},
        expected_return=result.expected_return,
        volatility=result.volatility,
        sharpe=result.sharpe if np.isfinite(result.sharpe) else 0.0,
        risk_contributions={k: float(v) for k, v in contributions.items()},
        diversification_ratio=float(diversification_ratio(result.weights.to_numpy(), covariance)),
        converged=result.converged,
    )


@router.post("/risk", response_model=dict)
def compute_risk(request: RiskRequest, session: Session = SessionDep) -> dict:
    """Risk summary for one asset over a trailing window."""
    prices = repo.load_prices(session, [request.symbol.upper()], as_of=request.as_of)
    if prices.empty:
        raise NotFoundError("no price data", symbol=request.symbol)

    series = prices.sort_values("ts")["adj_close"].astype("float64").tail(request.lookback_days + 1)
    returns = series.pct_change(fill_method=None).dropna()
    if len(returns) < 20:
        raise InsufficientDataError("not enough observations", have=int(len(returns)), need=20)

    equity = pd.Series((1.0 + returns).cumprod().to_numpy())
    summary = summarise_risk(returns, equity=equity).to_dict()
    return {
        "symbol": request.symbol.upper(),
        "as_of": request.as_of,
        "lookback_days": request.lookback_days,
        "confidence": request.confidence,
        **summary,
    }


__all__ = ["router"]

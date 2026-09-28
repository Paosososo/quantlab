"""Portfolio optimisation.

A deliberately small set of methods, with a warning attached
------------------------------------------------------------
Mean-variance optimisation is famously unstable: it needs expected returns,
those estimates are dominated by noise, and the optimiser responds by taking
enormous offsetting positions in assets whose estimated returns differ by an
amount smaller than the estimation error.  Michaud called it an
"error-maximising" procedure, and the criticism is fair.

The practical responses, all implemented here:

* **Minimum variance** needs no expected-return estimate at all, which removes
  the noisiest input.  It is usually the sensible default.
* **Risk parity** needs only the covariance and equalises risk *contributions*,
  which produces stable, diversified weights.
* **Shrunk covariance** (see :mod:`quantlab.risk.metrics`) conditions the one
  input that remains.
* **Weight bounds** cap what the optimiser can do with a bad estimate.

Maximum-Sharpe is provided because the comparison is instructive, not because it
is recommended.  Its docstring says so.

SLSQP from SciPy is used rather than a convex solver: the problems here are
small and smooth, SciPy is already a dependency, and adding cvxpy for a handful
of 10-asset optimisations is not worth the install.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantlab.exceptions import QuantlabError


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    method: str
    converged: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.round(6).to_dict(),
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "method": self.method,
            "converged": self.converged,
            "message": self.message,
        }


def _prepare(
    cov: pd.DataFrame, expected_returns: pd.Series | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    assets = list(cov.columns)
    matrix = np.asarray(cov.to_numpy(), dtype="float64")
    if matrix.shape[0] != matrix.shape[1]:
        raise QuantlabError("covariance matrix must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise QuantlabError("covariance matrix must be symmetric")
    mu = (
        np.asarray(expected_returns.reindex(assets).to_numpy(), dtype="float64")
        if expected_returns is not None
        else np.zeros(len(assets))
    )
    return matrix, mu, assets


def _summarise(
    weights: np.ndarray,
    cov: np.ndarray,
    mu: np.ndarray,
    assets: list[str],
    method: str,
    converged: bool,
    message: str,
    risk_free: float = 0.0,
) -> OptimisationResult:
    volatility = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    expected = float(weights @ mu)
    sharpe = (expected - risk_free) / volatility if volatility > 1e-12 else float("nan")
    return OptimisationResult(
        weights=pd.Series(weights, index=assets),
        expected_return=expected,
        volatility=volatility,
        sharpe=sharpe,
        method=method,
        converged=converged,
        message=message,
    )


def equal_weight(
    cov: pd.DataFrame, expected_returns: pd.Series | None = None
) -> OptimisationResult:
    """The benchmark every optimiser must beat.

    Equal weighting is hard to beat out of sample precisely because it uses no
    estimates and therefore has no estimation error (DeMiguel, Garlappi and
    Uppal, 2009).  Any optimiser that loses to it is not earning its complexity.
    """
    matrix, mu, assets = _prepare(cov, expected_returns)
    weights = np.full(len(assets), 1.0 / len(assets))
    return _summarise(weights, matrix, mu, assets, "equal_weight", True, "closed form")


def inverse_volatility(
    cov: pd.DataFrame, expected_returns: pd.Series | None = None
) -> OptimisationResult:
    """Weights proportional to 1/sigma.  Risk parity's diagonal approximation."""
    matrix, mu, assets = _prepare(cov, expected_returns)
    sigma = np.sqrt(np.diag(matrix))
    if np.any(sigma <= 0):
        raise QuantlabError("an asset has non-positive variance")
    weights = (1.0 / sigma) / np.sum(1.0 / sigma)
    return _summarise(weights, matrix, mu, assets, "inverse_volatility", True, "closed form")


def minimum_variance(
    cov: pd.DataFrame,
    *,
    expected_returns: pd.Series | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
    allow_short: bool = False,
) -> OptimisationResult:
    """Lowest-variance fully invested portfolio.

    The recommended default: no expected-return estimate means no
    error-maximisation, and empirically it tends to beat mean-variance out of
    sample for exactly that reason.
    """
    matrix, mu, assets = _prepare(cov, expected_returns)
    n = len(assets)
    limits = [(-abs(bounds[1]), abs(bounds[1])) if allow_short else bounds] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    result = minimize(
        lambda w: float(w @ matrix @ w),
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=limits,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return _summarise(
        np.asarray(result.x),
        matrix,
        mu,
        assets,
        "minimum_variance",
        bool(result.success),
        str(result.message),
    )


def maximum_sharpe(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    *,
    risk_free: float = 0.0,
    bounds: tuple[float, float] = (0.0, 1.0),
    allow_short: bool = False,
) -> OptimisationResult:
    """Tangency portfolio.

    Included for comparison rather than recommendation.  Its weights are highly
    sensitive to ``expected_returns``, which for financial assets are estimated
    with enormous error -- a one-standard-error change in an estimate can move a
    weight from 0 to the bound.  Compare its out-of-sample behaviour against
    :func:`minimum_variance` and :func:`equal_weight` before trusting it.
    """
    matrix, mu, assets = _prepare(cov, expected_returns)
    n = len(assets)
    limits = [(-abs(bounds[1]), abs(bounds[1])) if allow_short else bounds] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    def negative_sharpe(w: np.ndarray) -> float:
        volatility = float(np.sqrt(max(w @ matrix @ w, 1e-18)))
        return -float((w @ mu - risk_free) / volatility)

    result = minimize(
        negative_sharpe,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=limits,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return _summarise(
        np.asarray(result.x),
        matrix,
        mu,
        assets,
        "maximum_sharpe",
        bool(result.success),
        str(result.message),
        risk_free,
    )


def risk_parity(
    cov: pd.DataFrame,
    *,
    expected_returns: pd.Series | None = None,
    target_contributions: np.ndarray | None = None,
) -> OptimisationResult:
    """Equal (or specified) risk contributions.

    Asset ``i``'s risk contribution is ``w_i * (Sigma w)_i / sigma_p``, and these
    sum to portfolio volatility.  Equalising them gives the low-volatility assets
    larger weights, which is the intuition behind risk parity: a 60/40 portfolio
    is roughly 90% equity risk, and this fixes that.

    Long-only by construction; the objective is undefined for negative weights.
    """
    matrix, mu, assets = _prepare(cov, expected_returns)
    n = len(assets)
    targets = (
        np.full(n, 1.0 / n)
        if target_contributions is None
        else np.asarray(target_contributions, dtype="float64") / np.sum(target_contributions)
    )

    def objective(w: np.ndarray) -> float:
        portfolio_variance = float(w @ matrix @ w)
        if portfolio_variance <= 1e-18:
            return 1e6
        contributions = w * (matrix @ w) / np.sqrt(portfolio_variance)
        shares = contributions / np.sum(contributions)
        return float(np.sum((shares - targets) ** 2))

    with warnings.catch_warnings():
        # SLSQP reports when an intermediate step lands outside the bounds and
        # gets clipped.  That is the algorithm working as designed on a bounded
        # problem, not a failure; convergence is checked via ``result.success``.
        warnings.filterwarnings("ignore", message="Values in x were outside bounds")
        result = minimize(
            objective,
            x0=np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(1e-6, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"maxiter": 1000, "ftol": 1e-14},
        )
    weights = np.asarray(result.x)
    weights = weights / weights.sum()
    return _summarise(
        weights, matrix, mu, assets, "risk_parity", bool(result.success), str(result.message)
    )


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Each asset's share of portfolio volatility."""
    w = weights.reindex(cov.columns).to_numpy(dtype="float64")
    matrix = cov.to_numpy(dtype="float64")
    volatility = float(np.sqrt(max(w @ matrix @ w, 1e-18)))
    contributions = w * (matrix @ w) / volatility
    total = contributions.sum()
    return pd.Series(
        contributions / total if abs(total) > 1e-18 else contributions, index=cov.columns
    )


def efficient_frontier(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    *,
    n_points: int = 25,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Minimum-variance portfolio for a grid of target returns."""
    matrix, mu, assets = _prepare(cov, expected_returns)
    n = len(assets)
    grid = np.linspace(float(np.min(mu)), float(np.max(mu)), n_points)
    rows = []
    for target in grid:
        constraints = [
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu - t)},
        ]
        result = minimize(
            lambda w: float(w @ matrix @ w),
            x0=np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[bounds] * n,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            continue
        w = np.asarray(result.x)
        rows.append(
            {
                "target_return": float(target),
                "volatility": float(np.sqrt(max(w @ matrix @ w, 0.0))),
                "sharpe": float(target / np.sqrt(max(w @ matrix @ w, 1e-18))),
                **{asset: float(weight) for asset, weight in zip(assets, w, strict=True)},
            }
        )
    return pd.DataFrame(rows)


#: Name -> optimiser.  Typed explicitly so callers that dispatch by string
#: (the API, the dashboard) still type-check.
Optimiser = Callable[..., OptimisationResult]

OPTIMISERS: dict[str, Optimiser] = {
    "equal_weight": equal_weight,
    "inverse_volatility": inverse_volatility,
    "minimum_variance": minimum_variance,
    "risk_parity": risk_parity,
}


__all__ = [
    "OPTIMISERS",
    "OptimisationResult",
    "Optimiser",
    "efficient_frontier",
    "equal_weight",
    "inverse_volatility",
    "maximum_sharpe",
    "minimum_variance",
    "risk_contributions",
    "risk_parity",
]

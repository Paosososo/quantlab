"""Portfolio risk metrics.

Value at Risk is offered in three flavours because they disagree, and the
disagreement is the useful part:

**Historical** -- the empirical quantile of realised returns.  Makes no
distributional assumption, but cannot produce a loss larger than the worst one
already seen, so it understates tail risk on short samples.

**Parametric (normal)** -- mean plus z-quantile times standard deviation.
Convenient and wrong in a specific direction: daily equity returns have fat
tails, so a normal VaR understates the 1% loss, typically by 20-40%.

**Cornish-Fisher** -- adjusts the normal quantile for skewness and kurtosis.  A
middle ground that captures much of the fat-tail effect without needing a full
distributional model.

Conditional VaR (expected shortfall) is reported alongside every VaR because VaR
alone says nothing about how bad the bad days are, and is not subadditive:
combining two portfolios can raise total VaR, which makes it a poor risk measure
for aggregation.  CVaR does not have that flaw.

Sign convention: losses are reported as **negative** numbers throughout, matching
the return series they come from.  Mixing "VaR is 3%" with "VaR is -3%" across a
codebase is a classic source of sign errors, so one convention is enforced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

Method = Literal["historical", "parametric", "cornish_fisher"]


def _clean(returns: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(returns, dtype="float64")
    return values[np.isfinite(values)]


def value_at_risk(
    returns: np.ndarray | pd.Series,
    confidence: float = 0.95,
    method: Method = "historical",
) -> float:
    """Loss threshold exceeded with probability ``1 - confidence``.

    Returned as a negative number (a loss).
    """
    values = _clean(returns)
    if values.size < 20:
        return float("nan")
    tail = 1.0 - confidence

    if method == "historical":
        return float(np.quantile(values, tail))
    mean = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    z = float(stats.norm.ppf(tail))
    if method == "parametric":
        return mean + z * sigma
    if method == "cornish_fisher":
        skew = float(stats.skew(values))
        excess_kurt = float(stats.kurtosis(values))
        z_adjusted = (
            z
            + (z**2 - 1) * skew / 6.0
            + (z**3 - 3 * z) * excess_kurt / 24.0
            - (2 * z**3 - 5 * z) * skew**2 / 36.0
        )
        return mean + z_adjusted * sigma
    raise ValueError(f"unknown VaR method {method!r}")


def conditional_value_at_risk(
    returns: np.ndarray | pd.Series,
    confidence: float = 0.95,
    method: Method = "historical",
) -> float:
    """Mean loss conditional on breaching the VaR threshold (expected shortfall)."""
    values = _clean(returns)
    if values.size < 20:
        return float("nan")
    threshold = value_at_risk(values, confidence, method)
    tail_losses = values[values <= threshold]
    if tail_losses.size == 0:
        return float(threshold)
    return float(np.mean(tail_losses))


def rolling_var(
    returns: pd.Series, window: int = 252, confidence: float = 0.95, method: Method = "historical"
) -> pd.Series:
    """Trailing VaR.  Causal: each point uses only the preceding ``window`` returns."""
    return returns.rolling(window, min_periods=window).apply(
        lambda w: value_at_risk(w, confidence, method), raw=True
    )


def var_breach_rate(returns: pd.Series, var_series: pd.Series) -> float:
    """Share of days whose loss exceeded the VaR estimated *before* that day.

    The backtest of the risk model itself.  A 95% VaR should be breached about
    5% of the time; materially more means the model understates risk, materially
    less means it is too conservative and the portfolio is under-deployed.  The
    VaR series is shifted by one so the comparison uses an estimate that existed
    beforehand.
    """
    joined = pd.concat([returns, var_series.shift(1)], axis=1, join="inner").dropna()
    if joined.empty:
        return float("nan")
    return float((joined.iloc[:, 0] < joined.iloc[:, 1]).mean())


@dataclass(frozen=True, slots=True)
class RiskSummary:
    n: int
    annualised_volatility: float
    downside_volatility: float
    var_95_historical: float
    var_99_historical: float
    var_95_parametric: float
    var_95_cornish_fisher: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    skewness: float
    excess_kurtosis: float
    worst_day: float
    best_day: float

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in asdict(self).items()
        }


def summarise_risk(
    returns: pd.Series, *, periods_per_year: int = 252, equity: pd.Series | None = None
) -> RiskSummary:
    from quantlab.backtesting.metrics import max_drawdown

    values = _clean(returns)
    if values.size == 0:
        raise ValueError("no finite returns to summarise")
    downside = values[values < 0]
    curve = equity if equity is not None else (1.0 + pd.Series(values)).cumprod()
    return RiskSummary(
        n=int(values.size),
        annualised_volatility=float(np.std(values, ddof=1) * np.sqrt(periods_per_year)),
        downside_volatility=float(np.std(downside, ddof=1) * np.sqrt(periods_per_year))
        if downside.size > 1
        else float("nan"),
        var_95_historical=value_at_risk(values, 0.95, "historical"),
        var_99_historical=value_at_risk(values, 0.99, "historical"),
        var_95_parametric=value_at_risk(values, 0.95, "parametric"),
        var_95_cornish_fisher=value_at_risk(values, 0.95, "cornish_fisher"),
        cvar_95=conditional_value_at_risk(values, 0.95),
        cvar_99=conditional_value_at_risk(values, 0.99),
        max_drawdown=float(max_drawdown(pd.Series(curve))),
        skewness=float(stats.skew(values)) if values.size > 2 else float("nan"),
        excess_kurtosis=float(stats.kurtosis(values)) if values.size > 3 else float("nan"),
        worst_day=float(np.min(values)),
        best_day=float(np.max(values)),
    )


def covariance_matrix(returns: pd.DataFrame, *, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualised sample covariance."""
    clean = returns.dropna(how="all")
    return clean.cov() * periods_per_year


def shrunk_covariance(
    returns: pd.DataFrame, *, periods_per_year: int = 252, shrinkage: float | None = None
) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage towards a constant-correlation target.

    The sample covariance of N assets from T observations is badly conditioned
    when T is not much larger than N, and mean-variance optimisation amplifies
    exactly that error -- it puts the largest weights on the assets whose
    covariance is most underestimated.  Shrinking towards a structured target
    trades a little bias for a large reduction in variance and gives portfolios
    that are stable out of sample.

    ``shrinkage=None`` uses scikit-learn's analytical Ledoit-Wolf intensity.
    """
    from sklearn.covariance import LedoitWolf

    clean = returns.dropna()
    if clean.shape[0] < clean.shape[1] + 2:
        raise ValueError("need more observations than assets to estimate a covariance")
    if shrinkage is None:
        estimator = LedoitWolf().fit(clean.to_numpy())
        matrix = estimator.covariance_
    else:
        sample = np.cov(clean.to_numpy(), rowvar=False, ddof=1)
        target = np.diag(np.diag(sample))
        matrix = (1 - shrinkage) * sample + shrinkage * target
    return pd.DataFrame(matrix * periods_per_year, index=clean.columns, columns=clean.columns)


def correlation_from_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    deviations = np.sqrt(np.diag(cov.to_numpy()))
    outer = np.outer(deviations, deviations)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer > 0, cov.to_numpy() / outer, np.nan)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def diversification_ratio(weights: np.ndarray, cov: pd.DataFrame) -> float:
    """Weighted average volatility divided by portfolio volatility.

    1.0 means no diversification benefit; higher is better.  A useful check that
    an "optimised" portfolio is actually diversified rather than concentrated in
    whichever asset the estimator liked most.
    """
    w = np.asarray(weights, dtype="float64")
    sigma = np.sqrt(np.diag(cov.to_numpy()))
    portfolio_vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    if portfolio_vol < 1e-12:
        return float("nan")
    return float((w @ sigma) / portfolio_vol)


__all__ = [
    "Method",
    "RiskSummary",
    "conditional_value_at_risk",
    "correlation_from_covariance",
    "covariance_matrix",
    "diversification_ratio",
    "rolling_var",
    "shrunk_covariance",
    "summarise_risk",
    "value_at_risk",
    "var_breach_rate",
]

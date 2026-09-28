"""Hypothesis tests for comparing forecasts and strategies.

Three tools, each answering a question the backtest table cannot:

:func:`diebold_mariano`
    "Is model A's forecast error genuinely smaller than model B's, or is the
    difference sampling noise?"  This is the right test for the project's
    research question, because comparing two RMSE numbers without a test says
    nothing about whether the gap would survive on new data.

:func:`stationary_bootstrap_ci`
    "What is the confidence interval around this Sharpe ratio?"  The analytical
    interval assumes iid normal returns; the block bootstrap preserves
    autocorrelation and volatility clustering by resampling *blocks*, so the
    interval reflects the data rather than the assumption.  Typical answer for
    three years of daily data: the interval is roughly plus or minus one, which
    is a useful corrective to a reported Sharpe of 1.2.

:func:`paired_t_test`
    The simple parametric comparison, kept for completeness and for cases where
    its assumptions are defensible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

LossFunction = Literal["squared", "absolute"]


@dataclass(frozen=True, slots=True)
class TestOutcome:
    name: str
    statistic: float
    pvalue: float
    detail: dict[str, Any]

    def rejects(self, alpha: float = 0.05) -> bool:
        return bool(self.pvalue < alpha)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _newey_west_variance(values: np.ndarray, lags: int) -> float:
    """Long-run variance of a series with Bartlett weights."""
    n = len(values)
    centred = values - values.mean()
    gamma0 = float(np.dot(centred, centred) / n)
    total = gamma0
    for lag in range(1, lags + 1):
        if lag >= n:
            break
        gamma = float(np.dot(centred[lag:], centred[:-lag]) / n)
        weight = 1.0 - lag / (lags + 1.0)
        total += 2.0 * weight * gamma
    return max(total, 1e-300)


def diebold_mariano(
    errors_a: np.ndarray | pd.Series,
    errors_b: np.ndarray | pd.Series,
    *,
    horizon: int = 1,
    loss: LossFunction = "squared",
    small_sample_correction: bool = True,
) -> TestOutcome:
    """Test of equal predictive accuracy between two forecasts.

    Null hypothesis: the two forecasts have equal expected loss.  A negative
    statistic favours model A (smaller loss); positive favours model B.

    The Harvey-Leybourne-Newbold (1997) small-sample correction is applied by
    default and the statistic is compared against a t distribution rather than a
    normal, because the asymptotic version over-rejects badly on the sample
    sizes available here (a few hundred out-of-sample points).
    """
    a = np.asarray(errors_a, dtype="float64")
    b = np.asarray(errors_b, dtype="float64")
    if a.shape != b.shape:
        raise ValueError("error series must have the same shape")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 10:
        raise ValueError("need at least 10 paired observations for a Diebold-Mariano test")

    if loss == "squared":
        differential = a**2 - b**2
    else:
        differential = np.abs(a) - np.abs(b)

    mean_d = float(np.mean(differential))
    variance = _newey_west_variance(differential, max(0, horizon - 1))
    statistic = mean_d / np.sqrt(variance / n)

    if small_sample_correction:
        h = horizon
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        statistic *= float(np.sqrt(max(factor, 1e-12)))

    pvalue = 2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1))
    return TestOutcome(
        name="diebold_mariano",
        statistic=float(statistic),
        pvalue=float(pvalue),
        detail={
            "n": n,
            "horizon": horizon,
            "loss": loss,
            "mean_loss_differential": mean_d,
            "favours": "a" if mean_d < 0 else ("b" if mean_d > 0 else "neither"),
            "small_sample_correction": small_sample_correction,
        },
    )


def diebold_mariano_loss_differential(
    differential: np.ndarray | pd.Series,
    *,
    horizon: int = 1,
    hac_lags: int | None = None,
    small_sample_correction: bool = True,
) -> TestOutcome:
    """Test a chronological series of paired loss differences.

    The caller can first average asset-level differences within each date.  That
    makes dates, not correlated asset-date rows, the sample units.  HAC then
    accounts for serial dependence between the resulting daily observations.
    Negative differences favour the candidate forecast.
    """
    values = np.asarray(differential, dtype="float64")
    if values.ndim != 1:
        raise ValueError("loss differential must be one-dimensional")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 10:
        raise ValueError("need at least 10 paired dates for a Diebold-Mariano test")
    lags = (
        max(horizon - 1, int(np.floor(4 * (n / 100.0) ** (2 / 9))))
        if hac_lags is None
        else hac_lags
    )
    if not 0 <= lags < n:
        raise ValueError("hac_lags must be between zero and n - 1")

    mean_d = float(values.mean())
    variance = _newey_west_variance(values, lags)
    statistic = mean_d / np.sqrt(variance / n)
    if small_sample_correction:
        factor = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        statistic *= float(np.sqrt(max(factor, 1e-12)))
    pvalue = 2.0 * stats.t.sf(abs(statistic), df=n - 1)
    return TestOutcome(
        name="diebold_mariano_panel_daily",
        statistic=float(statistic),
        pvalue=float(pvalue),
        detail={
            "n": n,
            "horizon": horizon,
            "loss": "squared",
            "hac_lags": lags,
            "mean_loss_differential": mean_d,
            "favours": "a" if mean_d < 0 else ("b" if mean_d > 0 else "neither"),
            "small_sample_correction": small_sample_correction,
        },
    )


def paired_t_test(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> TestOutcome:
    x = np.asarray(a, dtype="float64")
    y = np.asarray(b, dtype="float64")
    mask = np.isfinite(x) & np.isfinite(y)
    result = stats.ttest_rel(x[mask], y[mask])
    return TestOutcome(
        name="paired_t_test",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        detail={"n": int(mask.sum()), "mean_difference": float(np.mean(x[mask] - y[mask]))},
    )


def mean_return_t_test(
    returns: np.ndarray | pd.Series, *, hac_lags: int | None = None
) -> TestOutcome:
    """Is the mean return different from zero, with HAC standard errors?"""
    values = np.asarray(returns, dtype="float64")
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 10:
        raise ValueError("need at least 10 observations")
    lags = hac_lags if hac_lags is not None else max(1, int(np.floor(4 * (n / 100.0) ** (2 / 9))))
    variance = _newey_west_variance(values, lags)
    statistic = float(np.mean(values) / np.sqrt(variance / n))
    pvalue = 2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1))
    return TestOutcome(
        name="mean_return_t_test",
        statistic=statistic,
        pvalue=float(pvalue),
        detail={"n": n, "mean": float(np.mean(values)), "hac_lags": lags},
    )


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis and Romano (1994) stationary bootstrap index draw.

    Block lengths are geometric with mean ``mean_block`` and the series wraps
    around, which keeps the resampled series stationary -- unlike a fixed-block
    bootstrap, whose edges break stationarity.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = 1.0 / max(mean_block, 1.0)
    indices = np.empty(n, dtype="int64")
    current = int(rng.integers(0, n))
    for i in range(n):
        indices[i] = current
        if rng.random() < p:
            current = int(rng.integers(0, n))
        else:
            current = (current + 1) % n
    return indices


def stationary_bootstrap_ci(
    values: np.ndarray | pd.Series,
    statistic_fn,
    *,
    n_boot: int = 1_000,
    mean_block: float = 20.0,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap confidence interval for any statistic of a time series.

    ``mean_block`` should be large enough to span the dependence in the data.
    For daily returns, 20 (about a month) is a reasonable default: long enough
    to preserve volatility clustering, short enough to give many distinct
    resamples.
    """
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    n = len(array)
    if n < 30:
        raise ValueError("need at least 30 observations to bootstrap")
    rng = np.random.default_rng(seed)
    point = float(statistic_fn(array))
    draws = np.empty(n_boot, dtype="float64")
    for b in range(n_boot):
        sample = array[stationary_bootstrap_indices(n, mean_block, rng)]
        draws[b] = float(statistic_fn(sample))
    finite = draws[np.isfinite(draws)]
    lower, upper = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return {
        "point_estimate": point,
        "lower": float(lower),
        "upper": float(upper),
        "alpha": alpha,
        "n_boot": int(len(finite)),
        "mean_block": mean_block,
        "bootstrap_std": float(np.std(finite, ddof=1)),
        "share_above_zero": float(np.mean(finite > 0)),
    }


def sharpe_statistic(periods_per_year: int = 252):
    """Return a Sharpe function suitable for :func:`stationary_bootstrap_ci`."""

    def compute(values: np.ndarray) -> float:
        sigma = float(np.std(values, ddof=1))
        if sigma < 1e-12:
            return float("nan")
        return float(np.mean(values) / sigma * np.sqrt(periods_per_year))

    return compute


__all__ = [
    "TestOutcome",
    "diebold_mariano",
    "mean_return_t_test",
    "paired_t_test",
    "sharpe_statistic",
    "stationary_bootstrap_ci",
    "stationary_bootstrap_indices",
]

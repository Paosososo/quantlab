"""Multiple testing and data snooping.

The problem this module exists for
----------------------------------
If you test 20 strategy variants at the 5% level, you expect one "significant"
result even when none of them work.  Backtesting is worse than that, because
every configuration choice -- lookback, threshold, rebalance frequency, universe
-- multiplies the number of effective trials, and most of them are never
reported.  A Sharpe ratio of 1.5 selected as the best of 100 attempts is not
evidence of skill; it is roughly what the best of 100 random strategies would
show.

Three corrections, in increasing sophistication:

**Bonferroni** -- multiply p-values by the number of tests.  Simple, always
valid, and very conservative when tests are correlated (which strategy variants
always are).

**Benjamini-Hochberg** -- control the *false discovery rate* instead of the
family-wise error rate.  The right choice for research screening: it accepts
that some discoveries will be false in exchange for finding more of the real
ones.

**Deflated Sharpe ratio** (Bailey and Lopez de Prado, 2014) -- asks directly:
given that I tried ``N`` configurations, and given this strategy's skewness,
kurtosis and track-record length, what is the probability its true Sharpe is
above zero?  It is the most honest single number this project can report about a
backtest, and it is usually sobering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# p-value corrections
# ---------------------------------------------------------------------------
def bonferroni(pvalues: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(pvalues, dtype="float64")
    return np.minimum(values * len(values), 1.0)


def benjamini_hochberg(pvalues: np.ndarray | pd.Series) -> np.ndarray:
    """BH step-up adjusted p-values (q-values).

    Implemented directly rather than imported so the monotonicity enforcement
    (the cumulative minimum applied from the largest p-value down) is visible;
    it is the step most hand-rolled implementations get wrong.
    """
    values = np.asarray(pvalues, dtype="float64")
    n = len(values)
    if n == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    scaled = ranked * n / np.arange(1, n + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(n, dtype="float64")
    adjusted[order] = np.minimum(monotone, 1.0)
    return adjusted


def adjust_pvalues(
    pvalues: dict[str, float] | pd.Series, method: str = "benjamini_hochberg"
) -> pd.DataFrame:
    series = pd.Series(pvalues, dtype="float64")
    if method == "bonferroni":
        adjusted = bonferroni(series.to_numpy())
    elif method == "benjamini_hochberg":
        adjusted = benjamini_hochberg(series.to_numpy())
    else:
        raise ValueError(f"unknown correction method {method!r}")
    return pd.DataFrame(
        {"p_value": series, "adjusted_p_value": adjusted, "method": method}
    ).sort_values("p_value")


# ---------------------------------------------------------------------------
# Sharpe ratio inference
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SharpeInference:
    observed_sharpe_annual: float
    observed_sharpe_per_period: float
    n_observations: int
    skewness: float
    kurtosis: float
    probabilistic_sharpe: float
    deflated_sharpe: float | None
    n_trials: int | None
    threshold_sharpe_per_period: float | None
    minimum_track_record_years: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _moments(returns: np.ndarray) -> tuple[float, float, float, int]:
    values = returns[np.isfinite(returns)]
    n = len(values)
    if n < 10:
        raise ValueError("need at least 10 observations")
    sigma = float(np.std(values, ddof=1))
    if sigma < 1e-15:
        raise ValueError("returns have no variance")
    sharpe = float(np.mean(values) / sigma)
    skew = float(stats.skew(values))
    # Non-excess kurtosis: the PSR formula uses the fourth standardised moment,
    # which is 3 for a normal distribution.
    kurt = float(stats.kurtosis(values, fisher=False))
    return sharpe, skew, kurt, n


def probabilistic_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    benchmark_sharpe_per_period: float = 0.0,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark_sharpe_per_period``.

    Bailey and Lopez de Prado (2012).  Corrects for track-record length and for
    non-normality: negative skewness and fat tails both reduce the probability,
    which is why an options-selling strategy with a good Sharpe scores poorly
    here.

    ``benchmark_sharpe_per_period`` is in the same frequency as ``returns``.
    """
    values = np.asarray(returns, dtype="float64")
    sharpe, skew, kurt, n = _moments(values)
    denominator = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denominator <= 0:
        return float("nan")
    z = (sharpe - benchmark_sharpe_per_period) * np.sqrt(n - 1) / np.sqrt(denominator)
    return float(stats.norm.cdf(z))


def expected_maximum_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent random strategies.

    The Gumbel approximation from Bailey and Lopez de Prado (2014):

        E[max] = sqrt(V) * [ (1 - g) * z(1 - 1/N) + g * z(1 - 1/(N*e)) ]

    with ``g`` the Euler-Mascheroni constant and ``z`` the normal quantile
    function.  This is the bar a strategy must clear to be more than the best of
    N coin flips.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return 0.0
    if sharpe_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sharpe_variance) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    *,
    n_trials: int,
    sharpe_variance: float | None = None,
    trial_sharpes: np.ndarray | pd.Series | None = None,
) -> float:
    """Probability the strategy's true Sharpe is positive, after deflating for
    the number of configurations tried.

    Supply ``trial_sharpes`` (the per-period Sharpe of every variant tested) when
    you have them; the variance across trials is what sets the selection
    threshold.  Falling back to a supplied ``sharpe_variance`` is possible but
    the number should then be justified.
    """
    values = np.asarray(returns, dtype="float64")
    if trial_sharpes is not None:
        trials = np.asarray(trial_sharpes, dtype="float64")
        trials = trials[np.isfinite(trials)]
        variance = float(np.var(trials, ddof=1)) if len(trials) > 1 else 0.0
    elif sharpe_variance is not None:
        variance = float(sharpe_variance)
    else:
        raise ValueError("supply either trial_sharpes or sharpe_variance")
    threshold = expected_maximum_sharpe(n_trials, variance)
    return probabilistic_sharpe_ratio(values, threshold)


def minimum_track_record_length(
    returns: np.ndarray | pd.Series,
    *,
    benchmark_sharpe_per_period: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed for the Sharpe to be significant at ``confidence``.

    Answers "how long would this track record have to be before I believed it?".
    For a strategy with a Sharpe of 0.5 the answer is usually several years,
    which is a useful thing to know before drawing conclusions from an 18-month
    backtest.

    Returns infinity when the observed Sharpe is at or below the benchmark: no
    amount of additional data makes a non-existent edge significant.  Callers
    that serialise to JSON will see null, since JSON has no infinity.
    """
    values = np.asarray(returns, dtype="float64")
    sharpe, skew, kurt, _n = _moments(values)
    gap = sharpe - benchmark_sharpe_per_period
    if gap <= 0:
        return float("inf")
    z = stats.norm.ppf(confidence)
    numerator = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    return float(1.0 + numerator * (z / gap) ** 2)


def analyse_sharpe(
    returns: np.ndarray | pd.Series,
    *,
    periods_per_year: int = 252,
    n_trials: int | None = None,
    trial_sharpes: np.ndarray | pd.Series | None = None,
    confidence: float = 0.95,
) -> SharpeInference:
    """Everything the module knows about one return series, in one object."""
    values = np.asarray(returns, dtype="float64")
    sharpe, skew, kurt, n = _moments(values)
    psr = probabilistic_sharpe_ratio(values, 0.0)

    deflated: float | None = None
    threshold: float | None = None
    if n_trials is not None and n_trials > 1:
        if trial_sharpes is not None:
            trials = np.asarray(trial_sharpes, dtype="float64")
            trials = trials[np.isfinite(trials)]
            variance = float(np.var(trials, ddof=1)) if len(trials) > 1 else 0.0
        else:
            variance = 0.0
        threshold = expected_maximum_sharpe(n_trials, variance)
        deflated = probabilistic_sharpe_ratio(values, threshold)

    try:
        mtrl_periods = minimum_track_record_length(values, confidence=confidence)
        mtrl_years = mtrl_periods / periods_per_year if np.isfinite(mtrl_periods) else float("inf")
    except ValueError:
        mtrl_years = float("nan")

    return SharpeInference(
        observed_sharpe_annual=sharpe * float(np.sqrt(periods_per_year)),
        observed_sharpe_per_period=sharpe,
        n_observations=n,
        skewness=skew,
        kurtosis=kurt,
        probabilistic_sharpe=psr,
        deflated_sharpe=deflated,
        n_trials=n_trials,
        threshold_sharpe_per_period=threshold,
        minimum_track_record_years=mtrl_years,
    )


__all__ = [
    "EULER_MASCHERONI",
    "SharpeInference",
    "adjust_pvalues",
    "analyse_sharpe",
    "benjamini_hochberg",
    "bonferroni",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "minimum_track_record_length",
    "probabilistic_sharpe_ratio",
]

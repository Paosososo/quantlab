"""Stationarity and autocorrelation diagnostics.

Why both ADF and KPSS
---------------------
They test opposite null hypotheses.  ADF's null is "there is a unit root"
(non-stationary); KPSS's null is "the series is stationary".  Running only one
conflates "evidence of stationarity" with "failure to reject non-stationarity",
which are not the same thing, especially on the short samples typical here.

The informative outcomes are the agreements:

* ADF rejects **and** KPSS does not reject  ->  stationary
* ADF does not reject **and** KPSS rejects  ->  unit root
* neither rejects                            ->  the sample is uninformative
* both reject                                ->  something odd, often a
  structural break or long memory

:func:`classify_stationarity` returns that four-way verdict rather than a single
p-value, because a single p-value here invites the wrong conclusion.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

Verdict = Literal["stationary", "unit_root", "inconclusive", "conflicting"]


@contextmanager
def _quiet_statsmodels():
    """Suppress statsmodels' return-type deprecation notices.

    Both ``adfuller`` and ``kpss`` warn that they will return a result object in
    a future release.  We unpack the current tuple deliberately, so the warning
    is noise for a caller; when the API changes, the unpacking will fail loudly
    and this wrapper is where the fix goes.  KPSS additionally warns when its
    p-value is clipped to the interpolation table, which is expected behaviour.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        yield


@dataclass(frozen=True, slots=True)
class TestResult:
    name: str
    statistic: float
    pvalue: float
    lags: int
    critical_values: dict[str, float]
    null_hypothesis: str

    def rejects(self, alpha: float = 0.05) -> bool:
        return self.pvalue < alpha

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def adf_test(series: pd.Series, *, regression: str = "c", autolag: str = "AIC") -> TestResult:
    values = pd.Series(series).dropna().astype("float64").to_numpy()
    with _quiet_statsmodels():
        stat, pvalue, lags, _nobs, criticals, _icbest = adfuller(
            values, regression=regression, autolag=autolag
        )
    return TestResult(
        name="augmented_dickey_fuller",
        statistic=float(stat),
        pvalue=float(pvalue),
        lags=int(lags),
        critical_values={k: float(v) for k, v in criticals.items()},
        null_hypothesis="a unit root is present (the series is non-stationary)",
    )


def kpss_test(series: pd.Series, *, regression: str = "c", nlags: str = "auto") -> TestResult:
    values = pd.Series(series).dropna().astype("float64").to_numpy()
    with _quiet_statsmodels():
        stat, pvalue, lags, criticals = kpss(values, regression=regression, nlags=nlags)
    return TestResult(
        name="kpss",
        statistic=float(stat),
        pvalue=float(pvalue),
        lags=int(lags),
        critical_values={k: float(v) for k, v in criticals.items()},
        null_hypothesis="the series is (trend) stationary",
    )


def classify_stationarity(series: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    adf = adf_test(series)
    kpss_result = kpss_test(series)
    adf_rejects = adf.rejects(alpha)
    kpss_rejects = kpss_result.rejects(alpha)

    verdict: Verdict
    if adf_rejects and not kpss_rejects:
        verdict = "stationary"
    elif not adf_rejects and kpss_rejects:
        verdict = "unit_root"
    elif not adf_rejects and not kpss_rejects:
        verdict = "inconclusive"
    else:
        verdict = "conflicting"

    return {
        "verdict": verdict,
        "alpha": alpha,
        "adf": adf.to_dict(),
        "kpss": kpss_result.to_dict(),
        "interpretation": {
            "stationary": "both tests agree the series is stationary",
            "unit_root": "both tests agree the series has a unit root",
            "inconclusive": "neither test rejects; the sample is too short or too noisy to tell",
            "conflicting": "both reject; suspect a structural break or long memory",
        }[verdict],
    }


def autocorrelation(series: pd.Series, nlags: int = 20) -> pd.DataFrame:
    """ACF and PACF with approximate 95% confidence bands.

    The bands are the usual +/- 1.96/sqrt(n), which assume white noise.  For
    returns with volatility clustering they are optimistic, so a coefficient
    that only just clears the band is not strong evidence.
    """
    values = pd.Series(series).dropna().astype("float64").to_numpy()
    if len(values) <= nlags + 1:
        raise ValueError("series is too short for the requested number of lags")
    acf_values = acf(values, nlags=nlags, fft=True)
    pacf_values = pacf(values, nlags=min(nlags, len(values) // 2 - 1))
    band = 1.96 / np.sqrt(len(values))
    size = min(len(acf_values), len(pacf_values))
    return pd.DataFrame(
        {
            "lag": np.arange(size),
            "acf": acf_values[:size],
            "pacf": pacf_values[:size],
            "lower_95": -band,
            "upper_95": band,
        }
    ).set_index("lag")


def ljung_box(series: pd.Series, lags: int = 10) -> pd.DataFrame:
    """Portmanteau test for autocorrelation up to ``lags``.

    Applied to raw returns it tests predictability of the level; applied to
    squared returns it tests volatility clustering, which is almost always
    present and is a useful sanity check that the data is real market-like data.
    """
    values = pd.Series(series).dropna().astype("float64")
    return acorr_ljungbox(values, lags=list(range(1, lags + 1)), return_df=True)


def volatility_clustering(series: pd.Series, lags: int = 10) -> dict[str, Any]:
    """Ljung-Box on squared returns: the standard ARCH-effect check."""
    squared = pd.Series(series).dropna().astype("float64") ** 2
    table = ljung_box(squared, lags=lags)
    smallest = float(table["lb_pvalue"].min())
    return {
        "min_pvalue": smallest,
        "clustering_detected": bool(smallest < 0.05),
        "table": table,
    }


__all__ = [
    "TestResult",
    "adf_test",
    "autocorrelation",
    "classify_stationarity",
    "kpss_test",
    "ljung_box",
    "volatility_clustering",
]

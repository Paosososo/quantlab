"""Regression with standard errors that survive financial data.

Ordinary OLS standard errors assume homoskedastic, independent errors.  Return
regressions violate both: volatility clusters, and overlapping forward returns
are mechanically autocorrelated.  Using textbook standard errors on overlapping
returns produces t-statistics that are inflated by roughly ``sqrt(h)`` for an
``h``-period horizon, which is how a great many "significant" predictors in the
literature were found.

So :func:`ols` reports Newey-West (HAC) standard errors by default, with the
lag chosen as ``h - 1`` when a horizon is supplied (the Hansen-Hodrick
prescription for overlapping observations) or by Newey and West's automatic
rule otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(slots=True)
class RegressionResult:
    params: pd.Series
    std_errors: pd.Series
    tstats: pd.Series
    pvalues: pd.Series
    r_squared: float
    adj_r_squared: float
    nobs: int
    cov_type: str
    hac_lags: int | None
    fitted: pd.Series
    residuals: pd.Series

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "coefficient": self.params,
                "std_error": self.std_errors,
                "t_stat": self.tstats,
                "p_value": self.pvalues,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": self.params.to_dict(),
            "std_errors": self.std_errors.to_dict(),
            "t_stats": self.tstats.to_dict(),
            "p_values": self.pvalues.to_dict(),
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "nobs": self.nobs,
            "cov_type": self.cov_type,
            "hac_lags": self.hac_lags,
        }

    def significant(self, alpha: float = 0.05) -> list[str]:
        return sorted(self.pvalues[self.pvalues < alpha].index)


def newey_west_lags(n: int) -> int:
    """Newey and West's automatic bandwidth: ``floor(4 * (n/100)^(2/9))``."""
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def ols(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    add_constant: bool = True,
    cov_type: str = "HAC",
    horizon: int | None = None,
    hac_lags: int | None = None,
) -> RegressionResult:
    """Fit ``y ~ X`` with heteroskedasticity- and autocorrelation-robust errors.

    Pass ``horizon`` when ``y`` is an overlapping ``h``-period return; the HAC
    lag is then set to ``h - 1``, which is the standard correction for the
    induced moving-average structure.
    """
    frame = pd.concat([y.rename("__y__"), X], axis=1, join="inner").dropna()
    if frame.empty:
        raise ValueError("no overlapping observations between y and X")
    target = frame["__y__"]
    design = frame.drop(columns=["__y__"])
    if add_constant:
        design = sm.add_constant(design, has_constant="add")

    kwargs: dict[str, Any] = {}
    resolved_lags: int | None = None
    if cov_type == "HAC":
        if hac_lags is not None:
            resolved_lags = hac_lags
        elif horizon is not None:
            resolved_lags = max(1, horizon - 1)
        else:
            resolved_lags = newey_west_lags(len(frame))
        kwargs = {"cov_type": "HAC", "cov_kwds": {"maxlags": resolved_lags, "use_correction": True}}
    elif cov_type != "nonrobust":
        kwargs = {"cov_type": cov_type}

    fitted = sm.OLS(target, design).fit(**kwargs)
    return RegressionResult(
        params=fitted.params,
        std_errors=fitted.bse,
        tstats=fitted.tvalues,
        pvalues=fitted.pvalues,
        r_squared=float(fitted.rsquared),
        adj_r_squared=float(fitted.rsquared_adj),
        nobs=int(fitted.nobs),
        cov_type=cov_type,
        hac_lags=resolved_lags,
        fitted=pd.Series(fitted.fittedvalues, index=frame.index),
        residuals=pd.Series(fitted.resid, index=frame.index),
    )


def univariate_screen(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    horizon: int | None = None,
) -> pd.DataFrame:
    """One simple regression per feature, ranked by |t|.

    Useful as a diagnostic, dangerous as a selection method: running k
    regressions and keeping the best is exactly the multiple-testing problem
    that :mod:`quantlab.research.multiple_testing` exists to correct, so the
    output carries raw p-values and the caller is expected to adjust them.
    """
    rows = []
    for column in X.columns:
        try:
            result = ols(y, X[[column]], horizon=horizon)
        except (ValueError, np.linalg.LinAlgError):
            continue
        rows.append(
            {
                "feature": column,
                "coefficient": float(result.params.get(column, np.nan)),
                "t_stat": float(result.tstats.get(column, np.nan)),
                "p_value": float(result.pvalues.get(column, np.nan)),
                "r_squared": result.r_squared,
                "nobs": result.nobs,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.reindex(table["t_stat"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """VIF per feature.  Above about 10 signals damaging multicollinearity.

    Matters here because momentum features at 21, 63, 126 and 252 days overlap
    heavily by construction, and collinear features make individual coefficients
    unstable even when the joint fit is fine.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    clean = X.dropna()
    if clean.shape[1] < 2 or clean.empty:
        return pd.Series(dtype="float64")
    design = sm.add_constant(clean, has_constant="add")
    values = design.to_numpy(dtype="float64")
    factors = {
        column: float(variance_inflation_factor(values, i))
        for i, column in enumerate(design.columns)
        if column != "const"
    }
    return pd.Series(factors).sort_values(ascending=False)


__all__ = [
    "RegressionResult",
    "newey_west_lags",
    "ols",
    "univariate_screen",
    "variance_inflation_factors",
]

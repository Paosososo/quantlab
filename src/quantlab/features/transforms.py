"""Causal transform primitives.

Every function here uses only information at or before each output point.  The
banned operations, and why:

``rolling(..., center=True)``
    Centres the window on the observation, so half of it is in the future.

``bfill`` / ``interpolate(limit_direction="both")``
    Fill a gap with a value that had not been published yet.

Whole-sample standardisation (``(x - x.mean()) / x.std()``)
    The mean and standard deviation are computed over the entire history,
    including the part after each observation.  This is the most common leak in
    student projects because it looks like harmless preprocessing.

``min_periods`` below the window length
    Produces a value from a partial window during warm-up.  That value is not
    wrong in a look-ahead sense, but it is a *different statistic* than the one
    the column claims to be, which quietly changes the feature's meaning in the
    first N rows.  We require full windows and leave NaN during warm-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.exceptions import LookAheadError


def _require_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} expects a Series")
    return series.astype("float64")


def lag(series: pd.Series, periods: int = 1) -> pd.Series:
    """Shift *backwards in information*: output at t equals input at t-periods."""
    if periods < 0:
        raise LookAheadError("negative lag would expose future values", periods=periods)
    return _require_series(series, "lag").shift(periods)


def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """Simple return over ``periods``.

    ``fill_method=None`` is passed explicitly.  Pandas' historical default was
    to forward-fill missing prices before differencing, which silently reports a
    0% return across a data gap instead of NaN.  A missing observation should
    propagate as missing, not as "the price did not move".
    """
    if periods < 1:
        raise LookAheadError("pct_change requires a positive lookback", periods=periods)
    return _require_series(series, "pct_change").pct_change(periods, fill_method=None)


def log_return(series: pd.Series, periods: int = 1) -> pd.Series:
    if periods < 1:
        raise LookAheadError("log_return requires a positive lookback", periods=periods)
    values = _require_series(series, "log_return")
    # Guard explicitly rather than relying on numpy's warning: a non-positive
    # price is bad data, and it should become NaN, not -inf.
    positive = values.where(values > 0.0)
    return np.log(positive).diff(periods)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return _require_series(series, "rolling_mean").rolling(window, min_periods=window).mean()


def rolling_std(series: pd.Series, window: int, ddof: int = 1) -> pd.Series:
    return _require_series(series, "rolling_std").rolling(window, min_periods=window).std(ddof=ddof)


def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return _require_series(series, "rolling_sum").rolling(window, min_periods=window).sum()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    return _require_series(series, "rolling_min").rolling(window, min_periods=window).min()


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    return _require_series(series, "rolling_max").rolling(window, min_periods=window).max()


def rolling_skew(series: pd.Series, window: int) -> pd.Series:
    return _require_series(series, "rolling_skew").rolling(window, min_periods=window).skew()


def rolling_kurtosis(series: pd.Series, window: int) -> pd.Series:
    return _require_series(series, "rolling_kurtosis").rolling(window, min_periods=window).kurt()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Standardise against the trailing window only.

    This is the safe replacement for whole-sample standardisation: at each
    point, the mean and standard deviation come from the preceding ``window``
    observations and nothing else.
    """
    values = _require_series(series, "rolling_zscore")
    mean = rolling_mean(values, window)
    std = rolling_std(values, window)
    return (values - mean) / std.replace(0.0, np.nan)


def ewm_mean(series: pd.Series, span: int) -> pd.Series:
    """Exponentially weighted mean.  ``adjust=False`` gives the recursive form,
    which depends only on past values; ``adjust=True`` is also causal but
    reweights the whole history at each step, which is harder to reason about
    for a streaming feature."""
    return _require_series(series, "ewm_mean").ewm(span=span, adjust=False, min_periods=span).mean()


def realised_volatility(returns: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
    """Annualised trailing standard deviation of returns."""
    return rolling_std(returns, window) * np.sqrt(periods_per_year)


def downside_volatility(returns: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
    """Annualised trailing standard deviation of negative returns only."""
    values = _require_series(returns, "downside_volatility")
    negative = values.where(values < 0.0, 0.0)
    return negative.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's relative strength index, computed causally.

    Wilder's original smoothing is an EWM with ``alpha = 1/window``; using a
    simple rolling mean gives a slightly different (and more common in code,
    less common in textbooks) variant.  We use Wilder's.
    """
    values = _require_series(series, "rsi")
    delta = values.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # The two degenerate cases have to be handled explicitly.  With no losses in
    # the window the ratio is infinite and RSI is 100 by definition; with no
    # movement at all it is 0/0, which carries no information, so it stays NaN
    # rather than being reported as the neutral 50 some implementations return.
    no_loss = avg_loss.eq(0.0)
    no_gain = avg_gain.eq(0.0)
    out = out.mask(no_loss & ~no_gain, 100.0)
    out = out.mask(no_gain & ~no_loss, 0.0)
    return out.mask(no_loss & no_gain, np.nan)


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Wilder's ATR.  Uses the previous close, never the next one."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def forward_return(series: pd.Series, horizon: int = 1) -> pd.Series:
    """Return from t to t+horizon.

    This is a **label**, not a feature.  It is deliberately in this module so
    that it is obvious when a pipeline uses it, and every call site must pass it
    to the label argument of the feature pipeline rather than the feature list.
    Using it as a predictor is the definition of target leakage.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    values = _require_series(series, "forward_return")
    return values.shift(-horizon) / values - 1.0


__all__ = [
    "average_true_range",
    "downside_volatility",
    "ewm_mean",
    "forward_return",
    "lag",
    "log_return",
    "pct_change",
    "realised_volatility",
    "rolling_kurtosis",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_skew",
    "rolling_std",
    "rolling_sum",
    "rolling_zscore",
    "rsi",
]

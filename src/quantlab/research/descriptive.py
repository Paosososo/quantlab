"""Descriptive statistics for return series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True, slots=True)
class DescriptiveStats:
    n: int
    mean: float
    std: float
    minimum: float
    q05: float
    median: float
    q95: float
    maximum: float
    skewness: float
    excess_kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    share_positive: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe(series: pd.Series) -> DescriptiveStats:
    """Summary statistics, including a normality test.

    Skewness and kurtosis are reported because they are the whole reason
    Sharpe-style metrics are incomplete for financial data: a strategy that sells
    options has a good Sharpe and a terrible left tail, and the mean/variance
    summary cannot tell you that.  The Jarque-Bera statistic almost always
    rejects normality for daily returns; it is included so the rejection is on
    the record rather than assumed.
    """
    values = pd.Series(series).dropna().astype("float64")
    if values.empty:
        raise ValueError("cannot describe an empty series")
    array = values.to_numpy()
    jb_stat, jb_p = stats.jarque_bera(array) if len(array) >= 8 else (float("nan"), float("nan"))
    return DescriptiveStats(
        n=int(len(array)),
        mean=float(np.mean(array)),
        std=float(np.std(array, ddof=1)) if len(array) > 1 else float("nan"),
        minimum=float(np.min(array)),
        q05=float(np.quantile(array, 0.05)),
        median=float(np.median(array)),
        q95=float(np.quantile(array, 0.95)),
        maximum=float(np.max(array)),
        skewness=float(stats.skew(array)) if len(array) > 2 else float("nan"),
        excess_kurtosis=float(stats.kurtosis(array)) if len(array) > 3 else float("nan"),
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pvalue=float(jb_p),
        share_positive=float(np.mean(array > 0)),
    )


def summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for every numeric column."""
    rows = []
    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series) or series.dropna().empty:
            continue
        rows.append({"series": column, **describe(series).to_dict()})
    return pd.DataFrame(rows).set_index("series") if rows else pd.DataFrame()


def correlation_matrix(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix.

    Spearman is often the better choice for financial features: a single crash
    day can dominate a Pearson correlation, and most of the questions asked of
    the matrix ("do these two features carry the same information?") are about
    monotone association rather than linear association.
    """
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError(f"unknown correlation method {method!r}")
    return frame.corr(method=method, numeric_only=True)


def drawdown_table(equity: pd.Series, top: int = 5) -> pd.DataFrame:
    """The ``top`` worst peak-to-trough episodes, with their dates and lengths."""
    equity = pd.Series(equity).dropna().astype("float64")
    if len(equity) < 3:
        return pd.DataFrame(columns=["peak", "trough", "recovery", "depth", "length"])
    running_max = equity.cummax()
    underwater = equity < running_max

    episodes: list[dict[str, Any]] = []
    start: Any = None
    for timestamp, flag in underwater.items():
        if flag and start is None:
            start = timestamp
        elif not flag and start is not None:
            episodes.append({"start": start, "end": timestamp})
            start = None
    if start is not None:
        episodes.append({"start": start, "end": equity.index[-1]})

    rows = []
    for episode in episodes:
        window = equity.loc[episode["start"] : episode["end"]]
        peak = float(running_max.loc[episode["start"]])
        trough_value = float(window.min())
        rows.append(
            {
                "peak": episode["start"],
                "trough": window.idxmin(),
                "recovery": episode["end"],
                "depth": trough_value / peak - 1.0,
                "length": int(len(window)),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("depth").head(top).reset_index(drop=True)


__all__ = [
    "DescriptiveStats",
    "correlation_matrix",
    "describe",
    "drawdown_table",
    "summary_table",
]

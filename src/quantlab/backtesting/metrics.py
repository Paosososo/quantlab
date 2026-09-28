"""Performance metrics.

Every definition here is written out rather than imported from a library,
because the details are where backtests disagree and an interviewer is entitled
to ask which convention you used.  The choices made, and why:

*Annualised return* is geometric (CAGR), not the arithmetic mean scaled up.  The
arithmetic mean overstates what an investor actually compounds whenever returns
are volatile.

*Annualised volatility* scales the per-period standard deviation by
``sqrt(periods_per_year)``.  That is exact only for independent returns.  Daily
equity returns have mild negative autocorrelation and strong volatility
clustering, so the scaled number is an approximation; it is the market
convention and is used here for comparability, with the caveat recorded.

*Sharpe* uses excess returns over a per-period risk-free rate.  With a zero
risk-free rate it reduces to the familiar mean/std form.  It is a sample
estimate with a wide confidence interval on short samples, which is why
:mod:`quantlab.research.multiple_testing` provides a deflated version.

*Sortino* divides by downside deviation measured against the same MAR used to
compute excess returns, counting only periods below it.  There is a competing
convention that averages squared shortfalls over *all* periods; that produces a
larger denominator and a smaller ratio.  We use the first; the docstring says so
because the two differ by enough to matter.

*Max drawdown* is computed on the equity curve, not on cumulative returns of a
resampled series, so it reflects the actual worst peak-to-trough loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantlab.timeutils import infer_periods_per_year

EPSILON = 1e-12


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def to_returns(equity: pd.Series) -> pd.Series:
    """Simple period returns from an equity curve."""
    equity = equity.astype("float64").dropna()
    # ``fill_method=None``: a missing equity point must not be forward-filled
    # into a fabricated 0% return, which would deflate volatility and inflate
    # every risk-adjusted ratio computed from it.
    return equity.pct_change(fill_method=None).dropna()


def total_return(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or abs(equity.iloc[0]) < EPSILON:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def years_elapsed(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return float("nan")
    span = index[-1] - index[0]
    return float(span.total_seconds() / (365.25 * 24 * 3600))


def annualised_return(equity: pd.Series, periods_per_year: int | None = None) -> float:
    """Geometric mean growth rate.

    Uses the wall-clock span when the index is a DatetimeIndex, which handles
    gaps and partial years correctly; falls back to observation count otherwise.
    """
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    growth = float(equity.iloc[-1] / equity.iloc[0])
    if isinstance(equity.index, pd.DatetimeIndex):
        years = years_elapsed(equity.index)
    else:
        ppy = periods_per_year or 252
        years = (len(equity) - 1) / ppy
    if not np.isfinite(years) or years <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def annualised_volatility(returns: pd.Series, periods_per_year: int) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int, risk_free_annual: float = 0.0) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    rf_period = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_period
    sigma = excess.std(ddof=1)
    if sigma < EPSILON:
        return float("nan")
    return float(excess.mean() / sigma * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int, risk_free_annual: float = 0.0
) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    mar_period = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = returns - mar_period
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf") if excess.mean() > 0 else float("nan")
    downside_deviation = float(np.sqrt((downside**2).mean()))
    if downside_deviation < EPSILON:
        return float("nan")
    return float(excess.mean() / downside_deviation * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    equity = equity.astype("float64").dropna()
    if equity.empty:
        return equity
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(dd.min()) if not dd.empty else float("nan")


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest run of consecutive observations spent below a previous peak."""
    dd = drawdown_series(equity)
    if dd.empty:
        return 0
    under = (dd < -EPSILON).to_numpy()
    longest = current = 0
    for flag in under:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def calmar_ratio(equity: pd.Series, periods_per_year: int | None = None) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd < EPSILON:
        return float("nan")
    return float(annualised_return(equity, periods_per_year) / mdd)


def win_rate(returns: pd.Series) -> float:
    returns = returns.dropna()
    non_zero = returns[returns.abs() > EPSILON]
    if non_zero.empty:
        return float("nan")
    return float((non_zero > 0).mean())


def profit_factor(pnl: pd.Series) -> float:
    """Gross profit divided by gross loss.  Above 1 means profitable in aggregate."""
    pnl = pnl.dropna()
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses < EPSILON:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def annualised_turnover(turnover: pd.Series, periods_per_year: int) -> float:
    """Mean per-period traded notional over equity, scaled to a year.

    One full portfolio replacement per year is 1.0 by this definition (a buy and
    a matching sell each count their own notional, so a strategy that fully
    rotates monthly reports roughly 24, not 12).  The convention is stated
    because turnover definitions differ by a factor of two across the industry.
    """
    turnover = turnover.dropna()
    if turnover.empty:
        return float("nan")
    return float(turnover.mean() * periods_per_year)


def average_exposure(gross_exposure: pd.Series, equity: pd.Series) -> float:
    joined = pd.concat([gross_exposure, equity], axis=1).dropna()
    if joined.empty:
        return float("nan")
    ratio = joined.iloc[:, 0] / joined.iloc[:, 1].replace(0.0, np.nan)
    return float(ratio.mean())


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------
def beta_alpha(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int
) -> tuple[float, float]:
    """OLS beta and annualised alpha of the strategy against the benchmark."""
    joined = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return float("nan"), float("nan")
    y = joined.iloc[:, 0].to_numpy()
    x = joined.iloc[:, 1].to_numpy()
    var = float(np.var(x, ddof=1))
    if var < EPSILON:
        return float("nan"), float("nan")
    beta = float(np.cov(y, x, ddof=1)[0, 1] / var)
    alpha_period = float(np.mean(y) - beta * np.mean(x))
    return beta, float((1.0 + alpha_period) ** periods_per_year - 1.0)


def tracking_error(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int
) -> float:
    joined = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return float("nan")
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int
) -> float:
    joined = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return float("nan")
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    sigma = active.std(ddof=1)
    if sigma < EPSILON:
        return float("nan")
    return float(active.mean() / sigma * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    periods: int
    periods_per_year: int
    years: float
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    annualised_turnover: float
    average_gross_exposure: float
    skewness: float
    excess_kurtosis: float
    best_period: float
    worst_period: float
    n_trades: int = 0
    trade_win_rate: float = float("nan")
    trade_profit_factor: float = float("nan")
    total_commission: float = 0.0
    total_slippage: float = 0.0
    benchmark_total_return: float = float("nan")
    benchmark_annualised_return: float = float("nan")
    benchmark_sharpe_ratio: float = float("nan")
    beta: float = float("nan")
    annualised_alpha: float = float("nan")
    tracking_error: float = float("nan")
    information_ratio: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {k: _json_safe(v) for k, v in asdict(self).items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def compute_metrics(
    equity: pd.Series,
    *,
    turnover: pd.Series | None = None,
    gross_exposure: pd.Series | None = None,
    benchmark_equity: pd.Series | None = None,
    trade_pnl: pd.Series | None = None,
    periods_per_year: int | None = None,
    risk_free_annual: float = 0.0,
    total_commission: float = 0.0,
    total_slippage: float = 0.0,
) -> PerformanceMetrics:
    equity = equity.astype("float64").dropna()
    ppy = periods_per_year or (
        infer_periods_per_year(equity.index) if isinstance(equity.index, pd.DatetimeIndex) else 252
    )
    returns = to_returns(equity)

    bench_return = bench_annual = bench_sharpe = float("nan")
    beta = alpha = te = ir = float("nan")
    if benchmark_equity is not None and not benchmark_equity.dropna().empty:
        bench = benchmark_equity.astype("float64").dropna()
        bench_return = total_return(bench)
        bench_annual = annualised_return(bench, ppy)
        bench_returns = to_returns(bench)
        bench_sharpe = sharpe_ratio(bench_returns, ppy, risk_free_annual)
        beta, alpha = beta_alpha(returns, bench_returns, ppy)
        te = tracking_error(returns, bench_returns, ppy)
        ir = information_ratio(returns, bench_returns, ppy)

    trades = trade_pnl.dropna() if trade_pnl is not None else pd.Series(dtype="float64")

    return PerformanceMetrics(
        periods=len(equity),
        periods_per_year=ppy,
        years=years_elapsed(equity.index)
        if isinstance(equity.index, pd.DatetimeIndex)
        else float("nan"),
        total_return=total_return(equity),
        annualised_return=annualised_return(equity, ppy),
        annualised_volatility=annualised_volatility(returns, ppy),
        sharpe_ratio=sharpe_ratio(returns, ppy, risk_free_annual),
        sortino_ratio=sortino_ratio(returns, ppy, risk_free_annual),
        max_drawdown=max_drawdown(equity),
        max_drawdown_duration=max_drawdown_duration(equity),
        calmar_ratio=calmar_ratio(equity, ppy),
        win_rate=win_rate(returns),
        profit_factor=profit_factor(returns),
        annualised_turnover=annualised_turnover(turnover, ppy)
        if turnover is not None
        else float("nan"),
        average_gross_exposure=average_exposure(gross_exposure, equity)
        if gross_exposure is not None
        else float("nan"),
        skewness=float(returns.skew()) if len(returns) > 2 else float("nan"),
        excess_kurtosis=float(returns.kurtosis()) if len(returns) > 3 else float("nan"),
        best_period=float(returns.max()) if not returns.empty else float("nan"),
        worst_period=float(returns.min()) if not returns.empty else float("nan"),
        n_trades=int(len(trades)),
        trade_win_rate=float((trades > 0).mean()) if not trades.empty else float("nan"),
        trade_profit_factor=profit_factor(trades) if not trades.empty else float("nan"),
        total_commission=float(total_commission),
        total_slippage=float(total_slippage),
        benchmark_total_return=bench_return,
        benchmark_annualised_return=bench_annual,
        benchmark_sharpe_ratio=bench_sharpe,
        beta=beta,
        annualised_alpha=alpha,
        tracking_error=te,
        information_ratio=ir,
    )


__all__ = [
    "EPSILON",
    "PerformanceMetrics",
    "annualised_return",
    "annualised_turnover",
    "annualised_volatility",
    "average_exposure",
    "beta_alpha",
    "calmar_ratio",
    "compute_metrics",
    "drawdown_series",
    "information_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "to_returns",
    "total_return",
    "tracking_error",
    "win_rate",
    "years_elapsed",
]

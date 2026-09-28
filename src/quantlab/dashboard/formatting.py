"""Pure presentation helpers.

Kept out of ``app.py`` so they can be unit tested without importing Streamlit,
which is awkward to test and slow to import.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

#: Metric key -> (label, formatter).  Ordered as a reader would want them.
METRIC_LAYOUT: list[tuple[str, str, str]] = [
    ("total_return", "Total return", "pct"),
    ("annualised_return", "Annualised return", "pct"),
    ("annualised_volatility", "Volatility", "pct"),
    ("sharpe_ratio", "Sharpe", "num"),
    ("sortino_ratio", "Sortino", "num"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("calmar_ratio", "Calmar", "num"),
    ("win_rate", "Win rate", "pct"),
    ("profit_factor", "Profit factor", "num"),
    ("annualised_turnover", "Turnover (annual)", "num"),
    ("n_trades", "Trades", "int"),
    ("total_commission", "Commission paid", "money"),
    ("total_slippage", "Slippage cost", "money"),
]


def format_value(value: Any, kind: str) -> str:
    """Format a metric, returning an em-free placeholder when it is missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if kind == "pct":
        return f"{float(value):.2%}"
    if kind == "money":
        return f"{float(value):,.0f}"
    if kind == "int":
        return f"{int(value):,}"
    return f"{float(value):.3f}"


def metrics_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, formatted value) pairs in display order, skipping absent metrics."""
    rows: list[tuple[str, str]] = []
    for key, label, kind in METRIC_LAYOUT:
        if key in metrics:
            rows.append((label, format_value(metrics.get(key), kind)))
    return rows


def normalise_curves(equity: pd.DataFrame) -> pd.DataFrame:
    """Rebase the strategy and benchmark to 100 at the first common date.

    Comparing raw equity levels is misleading whenever the two start from
    different capital; rebasing makes the chart answer the question a reader
    actually has, which is which line grew faster.
    """
    if equity.empty or "equity" not in equity.columns:
        return pd.DataFrame()
    out = equity.set_index("ts").sort_index()
    frame = pd.DataFrame(index=out.index)
    first = float(out["equity"].iloc[0])
    frame["Strategy"] = out["equity"] / first * 100.0 if first else out["equity"]
    if "benchmark_equity" in out.columns:
        bench = out["benchmark_equity"].astype("float64").dropna()
        if not bench.empty:
            frame["Benchmark"] = bench / float(bench.iloc[0]) * 100.0
    return frame


def drawdown_frame(equity: pd.DataFrame) -> pd.Series:
    if equity.empty or "equity" not in equity.columns:
        return pd.Series(dtype="float64")
    series = equity.set_index("ts").sort_index()["equity"].astype("float64")
    return series / series.cummax() - 1.0


def monthly_returns_table(equity: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month returns as a year x month grid."""
    if equity.empty or "equity" not in equity.columns:
        return pd.DataFrame()
    series = equity.set_index("ts").sort_index()["equity"].astype("float64")
    monthly = series.resample("ME").last().pct_change(fill_method=None).dropna()
    if monthly.empty:
        return pd.DataFrame()
    table = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.strftime("%b"),
            "ret": monthly.to_numpy(),
        }
    )
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = table.pivot(index="year", columns="month", values="ret")
    return pivot.reindex(columns=[m for m in order if m in pivot.columns])


def interpret_sharpe_inference(inference: dict[str, Any] | None) -> str:
    """One honest sentence about how much to trust a Sharpe ratio."""
    if not inference:
        return "Not enough observations to say anything about statistical significance."
    psr = inference.get("probabilistic_sharpe")
    dsr = inference.get("deflated_sharpe")
    trials = inference.get("n_trials")
    if psr is None:
        return "The Sharpe ratio could not be evaluated for significance."

    parts = [
        f"There is a {psr:.0%} probability the true Sharpe ratio is above zero, "
        "after adjusting for track-record length, skewness and fat tails."
    ]
    if dsr is not None and trials:
        parts.append(
            f"Deflating for the {trials} strategy configurations that have been "
            f"tested here lowers that to {dsr:.0%}."
        )
    years = inference.get("minimum_track_record_years")
    if years is None:
        parts.append(
            "No track record length would make this result significant, because the "
            "observed edge is not above the benchmark."
        )
    elif years > 0:
        parts.append(f"About {years:.1f} years of returns would be needed for 95% confidence.")
    return " ".join(parts)


__all__ = [
    "METRIC_LAYOUT",
    "drawdown_frame",
    "format_value",
    "interpret_sharpe_inference",
    "metrics_rows",
    "monthly_returns_table",
    "normalise_curves",
]

"""Close-only real-market forecast pilot, separate from the OHLCV backtest."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.models.registry import DEFAULT_REGRESSION_LADDER, get_model
from quantlab.models.splitters import ExpandingWindowSplitter, SplitterConfig
from quantlab.models.walkforward import WalkForwardRunner
from quantlab.research.hypothesis import diebold_mariano_loss_differential
from quantlab.research.multiple_testing import adjust_pvalues

SOURCE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"


def build_close_only_frame(
    source_csv: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    """Build causal close-based features for the following session's return.

    ``ts`` is a session label at 22:00 UTC, after the usual US market close.
    It is not a verified historical FRED publication timestamp.  The pilot
    evaluates forecasts only and must not feed a trading backtest.
    """
    raw = pd.read_csv(source_csv)
    if not {"observation_date", "SP500"}.issubset(raw.columns):
        raise ValueError("expected FRED columns observation_date and SP500")
    dates = pd.to_datetime(raw["observation_date"], errors="raise", utc=True)
    if dates.duplicated().any():
        raise ValueError("duplicate market dates")
    closes = pd.to_numeric(raw["SP500"], errors="coerce")
    valid = closes.notna() & np.isfinite(closes) & (closes > 0)
    panel = pd.DataFrame({"date": dates[valid], "close": closes[valid]}).sort_values("date")
    panel = panel.reset_index(drop=True)
    if len(panel) < 1_100:
        raise ValueError("not enough valid closes for the predeclared walk-forward study")

    ret_1d = panel["close"].pct_change(fill_method=None)
    X = pd.DataFrame(
        {
            "ret_1d": ret_1d,
            "ret_5d": panel["close"].pct_change(5, fill_method=None),
            "ret_21d": panel["close"].pct_change(21, fill_method=None),
            "vol_21d": ret_1d.rolling(21, min_periods=21).std(),
            "ma_gap_21d": panel["close"] / panel["close"].rolling(21).mean() - 1,
        }
    )
    y = (panel["close"].shift(-1) / panel["close"] - 1).rename("fwd_ret_1d")
    session_labels = panel["date"] + pd.Timedelta(hours=22)
    meta = pd.DataFrame(
        {
            "symbol": "SP500",
            "ts": session_labels,
            "available_at": session_labels,
            "label_available_at": session_labels.shift(-1),
        }
    )
    keep = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    meta = meta.loc[keep].reset_index(drop=True)
    summary = {
        "raw_rows": int(len(raw)),
        "valid_closes": int(len(panel)),
        "excluded_missing_or_invalid": int((~valid).sum()),
        "model_rows": int(len(X)),
        "first_close_date": str(panel["date"].iloc[0].date()),
        "last_close_date": str(panel["date"].iloc[-1].date()),
    }
    return X, y, meta, summary


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def run_close_only_pilot(source_csv: Path, output_dir: Path) -> dict[str, Any]:
    """Evaluate the fixed model ladder on real index closes, without trading."""
    source_csv = Path(source_csv)
    output_dir = Path(output_dir)
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing pilot: {output_dir}")
    X, y, meta, coverage = build_close_only_frame(source_csv)
    splitter = ExpandingWindowSplitter(
        SplitterConfig(
            n_splits=5,
            test_size=252,
            min_train_size=756,
            embargo=pd.Timedelta(days=2),
        )
    )
    runner = WalkForwardRunner(splitter)
    results = {
        name: runner.run(get_model(name, **params), X, y, meta, seed=20240101)
        for name, params in DEFAULT_REGRESSION_LADDER
    }
    baseline = results["zero"].predictions.set_index("ts").sort_index()
    forecast_rows: list[dict[str, Any]] = []
    significance_rows: list[dict[str, Any]] = []
    for name, _ in DEFAULT_REGRESSION_LADDER:
        result = results[name]
        assert result.overall is not None
        metric = result.overall
        forecast_rows.append(
            {
                "model": name,
                "n": metric.n,
                "rmse": metric.rmse,
                "r2_oos_vs_zero": metric.r2_oos_vs_zero,
                "directional_accuracy": _finite(metric.directional_accuracy),
                "information_coefficient": _finite(metric.information_coefficient),
            }
        )
        if name == "zero":
            continue
        candidate = result.predictions.set_index("ts").sort_index()
        paired = candidate.join(baseline[["y_true", "y_pred"]], rsuffix="_base", how="inner")
        if not np.allclose(paired["y_true"], paired["y_true_base"], rtol=0, atol=1e-12):
            raise ValueError(f"candidate and baseline targets differ for {name}")
        differential = (
            (paired["y_true"] - paired["y_pred"]) ** 2
            - (paired["y_true"] - paired["y_pred_base"]) ** 2
        ).sort_index()
        outcome = diebold_mariano_loss_differential(differential, horizon=1)
        significance_rows.append(
            {
                "model": name,
                "n_dates": outcome.detail["n"],
                "hac_lags": outcome.detail["hac_lags"],
                "mean_loss_differential": outcome.detail["mean_loss_differential"],
                "dm_statistic": outcome.statistic,
                "p_value": outcome.pvalue,
                "favours": outcome.detail["favours"],
            }
        )

    adjusted = adjust_pvalues(
        {row["model"]: row["p_value"] for row in significance_rows},
        method="benjamini_hochberg",
    )
    for row in significance_rows:
        row["adjusted_p_value"] = float(adjusted.loc[row["model"], "adjusted_p_value"])
        row["beats_zero_at_5pct"] = row["favours"] == "a" and row["adjusted_p_value"] < 0.05

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / "SP500_fred_source.csv"
    shutil.copyfile(source_csv, snapshot)
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    payload = {
        "study": "S&P 500 close-only forecast pilot",
        "source": "FRED SP500, S&P Dow Jones Indices LLC",
        "source_url": SOURCE_URL,
        "source_sha256": digest,
        "coverage": coverage,
        "features": list(X.columns),
        "target": "next observed session close-to-close price return",
        "splitter": splitter.describe(),
        "folds": [fold.split for fold in results["zero"].folds],
        "forecast_accuracy": forecast_rows,
        "statistical_significance": significance_rows,
        "limitations": [
            "Index levels are price-only and omit dividends.",
            "FRED supplies closes only; no OHLCV execution backtest or trading-cost estimate was run.",
            "The 22:00 UTC session label is not a verified historical FRED publication timestamp.",
            "One index and one historical sample do not establish a persistent market edge.",
        ],
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    forecast_table = pd.DataFrame(forecast_rows).round(6).to_markdown(index=False)
    display_significance = pd.DataFrame(significance_rows)
    for column in ("mean_loss_differential", "p_value", "adjusted_p_value"):
        display_significance[column] = display_significance[column].map(
            lambda value: f"{value:.3g}"
        )
    display_significance["dm_statistic"] = display_significance["dm_statistic"].map(
        lambda value: f"{value:.4f}"
    )
    significance_table = display_significance.to_markdown(index=False)
    report = "\n".join(
        [
            "# Real-market forecast pilot: S&P 500",
            "",
            "This is a close-only forecast experiment, not a trading backtest.",
            f"Source: {SOURCE_URL}",
            f"Raw CSV SHA-256: `{digest}`",
            f"Observed closes: {coverage['first_close_date']} to {coverage['last_close_date']}",
            f"Valid closes: {coverage['valid_closes']}; model rows: {coverage['model_rows']}.",
            "",
            "Five chronological expanding folds, 252 test sessions each, with a two-day embargo.",
            "The target is the next observed session's close-to-close price return.",
            "Features use the current and earlier closes only.",
            "",
            "## Forecast accuracy",
            "",
            forecast_table,
            "",
            "## Statistical comparison with a zero-return forecast",
            "",
            "Paired squared-loss differences are ordered by date. The test uses HAC standard",
            "errors and Benjamini-Hochberg adjusted p-values across the six comparisons.",
            "A negative statistic favours the candidate model.",
            "",
            significance_table,
            "",
            "## Limits",
            "",
            *[f"- {item}" for item in payload["limitations"]],
            "",
        ]
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return {"report": str(report_path), "results": str(result_path), "source": str(snapshot)}


__all__ = ["SOURCE_URL", "build_close_only_frame", "run_close_only_pilot"]

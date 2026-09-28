"""Walk-forward evaluation.

For each fold the runner takes a *fresh* model, fits it on the training rows and
predicts the test rows.  Three details make the resulting predictions safe to
feed into a backtest:

1.  **A new model per fold** (``model.clone()``).  Reusing a fitted object would
    carry state across the fold boundary.
2.  **Preprocessing inside the model.**  The scaler lives in the model's
    pipeline, so it is fitted on training rows only.  Scaling before splitting
    is a leak that is almost invisible in the metrics.
3.  **Honest availability on every prediction.**  Each output row carries
    ``available_at``, the availability of the features that produced it, so the
    backtester can filter on it exactly as it filters price data.  A prediction
    cannot be acted on before the information that produced it existed.

The runner also records the training-set mean per fold, because the historical
mean benchmark for out-of-sample R-squared must come from the training window,
not from the test window.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.exceptions import InsufficientDataError
from quantlab.features.guards import assert_split_ordering
from quantlab.logging import get_logger
from quantlab.models.base import Model
from quantlab.models.evaluation import ForecastMetrics, evaluate_forecasts
from quantlab.models.splitters import Split, TimeSeriesSplitter

log = get_logger(__name__)

REQUIRED_META_COLUMNS = ("symbol", "ts", "available_at")


@dataclass(slots=True)
class FoldResult:
    fold: int
    split: dict[str, Any]
    metrics: ForecastMetrics
    train_mean: float
    fit_seconds: float
    feature_importance: dict[str, float] | None = None


@dataclass(slots=True)
class WalkForwardResult:
    model_name: str
    model_params: dict[str, Any]
    splitter: dict[str, Any]
    predictions: pd.DataFrame
    folds: list[FoldResult] = field(default_factory=list)
    overall: ForecastMetrics | None = None
    target_name: str = ""
    seed: int = 0
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "params": self.model_params,
            "splitter": self.splitter,
            "target": self.target_name,
            "n_folds": len(self.folds),
            "n_predictions": int(len(self.predictions)),
            "n_features": self.n_features,
            "seed": self.seed,
            "overall": self.overall.to_dict() if self.overall else None,
            "folds": [
                {"fold": f.fold, **f.split, "metrics": f.metrics.to_dict()} for f in self.folds
            ],
        }

    def fold_metrics_frame(self) -> pd.DataFrame:
        if not self.folds:
            return pd.DataFrame()
        rows = [{"fold": f.fold, **f.metrics.to_dict()} for f in self.folds]
        return pd.DataFrame(rows).set_index("fold")

    def mean_feature_importance(self) -> pd.Series:
        """Average absolute importance across folds.

        Averaging matters: a feature that looks essential in one fold and
        irrelevant in the next is telling you the relationship is unstable, and
        that is worth seeing rather than reading a single fold's ranking.
        """
        frames = [f.feature_importance for f in self.folds if f.feature_importance]
        if not frames:
            return pd.Series(dtype="float64")
        table = pd.DataFrame(frames).abs()
        return table.mean(axis=0).sort_values(ascending=False)


class WalkForwardRunner:
    def __init__(
        self,
        splitter: TimeSeriesSplitter,
        *,
        min_test_rows: int = 5,
        collect_importance: bool = True,
    ) -> None:
        self.splitter = splitter
        self.min_test_rows = min_test_rows
        self.collect_importance = collect_importance

    def run(
        self,
        model: Model,
        X: pd.DataFrame,
        y: pd.Series,
        meta: pd.DataFrame,
        *,
        seed: int = 0,
    ) -> WalkForwardResult:
        missing = [c for c in REQUIRED_META_COLUMNS if c not in meta.columns]
        if missing:
            raise ValueError(f"meta frame is missing columns: {missing}")
        if not (len(X) == len(y) == len(meta)):
            raise InsufficientDataError(
                "X, y and meta must be aligned", n_x=len(X), n_y=len(y), n_meta=len(meta)
            )

        prediction_rows: list[pd.DataFrame] = []
        folds: list[FoldResult] = []

        for split in self.splitter.split(meta):
            if len(split.test_index) < self.min_test_rows:
                log.warning("walkforward.fold_too_small", fold=split.fold, n=len(split.test_index))
                continue

            train_times = pd.DatetimeIndex(
                pd.to_datetime(meta["ts"].iloc[split.train_index], utc=True)
            )
            test_times = pd.DatetimeIndex(
                pd.to_datetime(meta["ts"].iloc[split.test_index], utc=True)
            )
            assert_split_ordering(train_times, test_times, label=f"fold_{split.fold}")

            fold_result = self._run_fold(model, X, y, meta, split, prediction_rows)
            folds.append(fold_result)

        if not folds:
            raise InsufficientDataError("no usable folds", splitter=self.splitter.name, rows=len(X))

        predictions = pd.concat(prediction_rows, ignore_index=True).sort_values(["ts", "symbol"])
        overall = evaluate_forecasts(
            predictions["y_true"].to_numpy(),
            predictions["y_pred"].to_numpy(),
            y_proba=predictions["y_proba"].to_numpy() if "y_proba" in predictions else None,
            train_mean=float(np.mean([f.train_mean for f in folds])),
            task=model.task,
        )
        log.info(
            "walkforward.complete",
            model=model.name,
            folds=len(folds),
            predictions=len(predictions),
            r2_oos=round(overall.r2_oos_vs_zero, 6),
            ic=round(overall.information_coefficient, 4)
            if np.isfinite(overall.information_coefficient)
            else None,
        )
        return WalkForwardResult(
            model_name=model.name,
            model_params=model.get_params(),
            splitter=self.splitter.describe(),
            predictions=predictions.reset_index(drop=True),
            folds=folds,
            overall=overall,
            target_name=str(y.name or "target"),
            seed=seed,
            n_features=X.shape[1],
            feature_names=list(X.columns),
        )

    def _run_fold(
        self,
        model: Model,
        X: pd.DataFrame,
        y: pd.Series,
        meta: pd.DataFrame,
        split: Split,
        sink: list[pd.DataFrame],
    ) -> FoldResult:
        X_train = X.iloc[split.train_index]
        y_train = y.iloc[split.train_index]
        X_test = X.iloc[split.test_index]
        y_test = y.iloc[split.test_index]
        meta_test = meta.iloc[split.test_index]

        # A brand-new model per fold: no state crosses the boundary.
        fold_model = model.clone()
        started = time.perf_counter()
        fold_model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        predictions = fold_model.predict(X_test)
        probabilities = fold_model.predict_proba(X_test)
        train_mean = float(np.mean(y_train.to_numpy()))

        frame = pd.DataFrame(
            {
                "symbol": meta_test["symbol"].to_numpy(),
                "ts": pd.to_datetime(meta_test["ts"], utc=True).to_numpy(),
                # The forecast is knowable exactly when its inputs were.
                "available_at": pd.to_datetime(meta_test["available_at"], utc=True).to_numpy(),
                "fold": split.fold,
                "y_pred": predictions,
                "y_true": y_test.to_numpy(),
            }
        )
        if "label_available_at" in meta_test.columns:
            frame["target_ts"] = pd.to_datetime(
                meta_test["label_available_at"], utc=True
            ).to_numpy()
        if probabilities is not None:
            frame["y_proba"] = probabilities
        sink.append(frame)

        importance = None
        if self.collect_importance:
            series = fold_model.feature_importance()
            if series is not None:
                importance = {str(k): float(v) for k, v in series.items()}

        return FoldResult(
            fold=split.fold,
            split=split.describe(),
            metrics=evaluate_forecasts(
                y_test.to_numpy(),
                predictions,
                y_proba=probabilities,
                train_mean=train_mean,
                task=fold_model.task,
            ),
            train_mean=train_mean,
            fit_seconds=fit_seconds,
            feature_importance=importance,
        )


def compare_models(
    specs: Sequence[tuple[str, dict[str, Any]]],
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    splitter: TimeSeriesSplitter,
    *,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, WalkForwardResult]]:
    """Run the whole ladder of models on identical folds.

    Identical folds is the point: comparing a model evaluated on one split
    against another evaluated on a different split says nothing.
    """
    from quantlab.models.registry import get_model

    runner = WalkForwardRunner(splitter)
    results: dict[str, WalkForwardResult] = {}
    rows: list[dict[str, Any]] = []
    for name, params in specs:
        model = get_model(name, **params)
        try:
            result = runner.run(model, X, y, meta, seed=seed)
        except (InsufficientDataError, ValueError) as exc:
            log.error("compare_models.failed", model=name, error=str(exc))
            continue
        key = f"{name}"
        suffix = 1
        while key in results:
            suffix += 1
            key = f"{name}_{suffix}"
        results[key] = result
        assert result.overall is not None
        rows.append({"model": key, **result.overall.to_dict()})
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame(), results


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


__all__ = [
    "FoldResult",
    "WalkForwardResult",
    "WalkForwardRunner",
    "compare_models",
]

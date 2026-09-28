"""Forecast evaluation metrics.

The one that matters most is :func:`out_of_sample_r2`.

Ordinary R-squared compares a model against the mean of the *test* sample, which
is a quantity nobody knew in advance.  Campbell and Thompson (2008) define the
out-of-sample R-squared against a benchmark forecast that was actually available:

    R2_OS = 1 - MSE(model) / MSE(benchmark)

with the benchmark usually the historical mean, or zero for excess returns.  For
monthly equity returns, values above about 0.005 are considered economically
meaningful in that literature; anything above 0.05 on daily data should be
treated as a bug until proven otherwise, and the first thing to check is
look-ahead bias.

Also included is the **information coefficient**, the correlation between
forecast and outcome.  Practitioners quote it because it maps directly onto
achievable Sharpe through the fundamental law of active management, and because
a model can have a useless R-squared and still be tradable if it gets the
*ranking* right.  The Spearman version is reported alongside Pearson since
financial returns are heavy-tailed enough that a handful of observations can
dominate a Pearson correlation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats

EPSILON = 1e-12


def _clean(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(true) & np.isfinite(pred)
    return true[mask], pred[mask]


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def mean_squared_error(y_true, y_pred) -> float:
    true, pred = _clean(y_true, y_pred)
    return float(np.mean((true - pred) ** 2)) if true.size else float("nan")


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred) -> float:
    true, pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(true - pred))) if true.size else float("nan")


def out_of_sample_r2(y_true, y_pred, benchmark=None) -> float:
    """Campbell-Thompson out-of-sample R-squared.

    ``benchmark`` defaults to a zero forecast, the right null for excess
    returns.  A negative value means the model is worse than predicting nothing,
    which is the usual honest outcome on daily data.
    """
    true, pred = _clean(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    reference = (
        np.zeros_like(true)
        if benchmark is None
        else np.asarray(benchmark, dtype="float64")[: true.size]
    )
    numerator = float(np.sum((true - pred) ** 2))
    denominator = float(np.sum((true - reference) ** 2))
    if denominator < EPSILON:
        return float("nan")
    return 1.0 - numerator / denominator


def directional_accuracy(y_true, y_pred) -> float:
    """Share of forecasts whose sign matches the outcome.

    Zero forecasts are excluded: a model that always predicts exactly zero has
    no directional view, and counting those as ties would flatter it.
    """
    true, pred = _clean(y_true, y_pred)
    mask = np.abs(pred) > EPSILON
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(true[mask]) == np.sign(pred[mask])))


def information_coefficient(y_true, y_pred) -> float:
    true, pred = _clean(y_true, y_pred)
    if true.size < 3 or np.std(pred) < EPSILON or np.std(true) < EPSILON:
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def rank_information_coefficient(y_true, y_pred) -> float:
    true, pred = _clean(y_true, y_pred)
    if true.size < 3 or np.std(pred) < EPSILON:
        return float("nan")
    result = stats.spearmanr(true, pred)
    return float(result.statistic)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def accuracy(y_true, y_pred) -> float:
    true, pred = _clean(y_true, y_pred)
    if true.size == 0:
        return float("nan")
    return float(np.mean((pred >= 0.5).astype(int) == (true >= 0.5).astype(int)))


def roc_auc(y_true, y_score) -> float:
    """Area under the ROC curve, via the Mann-Whitney U identity.

    Implemented directly rather than imported so the definition is visible and
    so the module has one fewer import; it agrees with scikit-learn to floating
    point on tied and untied data.
    """
    true, score = _clean(y_true, y_score)
    labels = (true >= 0.5).astype(int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def log_loss(y_true, y_proba, eps: float = 1e-15) -> float:
    true, proba = _clean(y_true, y_proba)
    if true.size == 0:
        return float("nan")
    labels = (true >= 0.5).astype("float64")
    clipped = np.clip(proba, eps, 1 - eps)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def brier_score(y_true, y_proba) -> float:
    true, proba = _clean(y_true, y_proba)
    if true.size == 0:
        return float("nan")
    return float(np.mean(((true >= 0.5).astype("float64") - proba) ** 2))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    n: int
    rmse: float
    mae: float
    r2_oos_vs_zero: float
    r2_oos_vs_mean: float
    directional_accuracy: float
    information_coefficient: float
    rank_information_coefficient: float
    mean_prediction: float
    std_prediction: float
    accuracy: float = float("nan")
    roc_auc: float = float("nan")
    log_loss: float = float("nan")
    brier_score: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in asdict(self).items()
        }


def evaluate_forecasts(
    y_true,
    y_pred,
    *,
    y_proba=None,
    train_mean: float | None = None,
    task: str = "regression",
) -> ForecastMetrics:
    """Compute every applicable metric.

    ``train_mean`` is the mean of the *training* targets, used as the historical
    mean benchmark.  Passing the test-set mean instead would be look-ahead bias
    inside the evaluation itself, which is a real and easily missed mistake.
    """
    true, pred = _clean(y_true, y_pred)
    mean_benchmark = (
        np.full_like(true, float(train_mean)) if train_mean is not None else np.full_like(true, 0.0)
    )
    metrics: dict[str, Any] = {
        "n": int(true.size),
        "rmse": root_mean_squared_error(true, pred),
        "mae": mean_absolute_error(true, pred),
        "r2_oos_vs_zero": out_of_sample_r2(true, pred),
        "r2_oos_vs_mean": out_of_sample_r2(true, pred, mean_benchmark),
        "directional_accuracy": directional_accuracy(true, pred),
        "information_coefficient": information_coefficient(true, pred),
        "rank_information_coefficient": rank_information_coefficient(true, pred),
        "mean_prediction": float(np.mean(pred)) if pred.size else float("nan"),
        "std_prediction": float(np.std(pred, ddof=1)) if pred.size > 1 else float("nan"),
    }
    if task == "classification":
        probabilities = y_proba if y_proba is not None else pred
        metrics.update(
            {
                "accuracy": accuracy(true, pred),
                "roc_auc": roc_auc(true, probabilities),
                "log_loss": log_loss(true, probabilities),
                "brier_score": brier_score(true, probabilities),
            }
        )
    return ForecastMetrics(**metrics)


__all__ = [
    "ForecastMetrics",
    "accuracy",
    "brier_score",
    "directional_accuracy",
    "evaluate_forecasts",
    "information_coefficient",
    "log_loss",
    "mean_absolute_error",
    "mean_squared_error",
    "out_of_sample_r2",
    "rank_information_coefficient",
    "roc_auc",
    "root_mean_squared_error",
]

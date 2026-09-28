"""Naive baselines.

These exist because the project's research question is whether machine learning
beats simple alternatives, and that question is only meaningful if the simple
alternatives are implemented honestly and evaluated identically.

``ZeroForecast`` is the important one.  For daily equity returns, predicting
zero is close to optimal in mean-squared-error terms, because the conditional
mean is tiny relative to the noise.  A model that cannot beat it on RMSE is not
finding anything, and a great many published "ML beats the market" results
quietly omit this comparison.
"""

from __future__ import annotations

import numpy as np

from quantlab.models.base import Model, Task


class ZeroForecast(Model):
    """Always predicts zero.

    The correct null hypothesis for return prediction under the efficient market
    view: tomorrow's excess return is unforecastable, so the best guess is no
    move.
    """

    name = "zero"
    task: Task = "regression"
    min_train_samples = 1

    def _fit(self, X, y, sample_weight) -> None:  # noqa: ARG002
        return None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype="float64")


class HistoricalMeanForecast(Model):
    """Predicts the mean of the training targets.

    Slightly stronger than zero when the asset has a positive drift, which is
    exactly the point: it separates "the model found a signal" from "the model
    discovered that stocks go up".
    """

    name = "historical_mean"
    task: Task = "regression"
    min_train_samples = 5

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.mean_: float = 0.0

    def _fit(self, X, y, sample_weight) -> None:  # noqa: ARG002
        self.mean_ = float(np.average(y, weights=sample_weight))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype="float64")


class LastValueForecast(Model):
    """Predicts the most recent value of a named feature.

    With ``feature="ret_1d"`` this is the random-walk-in-returns forecast: a
    momentum-of-one-day baseline that is surprisingly hard to beat at short
    horizons, and a good check that a model is doing more than reproducing the
    most recent observation.
    """

    name = "last_value"
    task: Task = "regression"
    min_train_samples = 1

    def __init__(self, feature: str = "ret_1d", **params) -> None:
        super().__init__(feature=feature, **params)
        self.feature = feature
        self._column: int | None = None

    def _fit(self, X, y, sample_weight) -> None:  # noqa: ARG002
        if self.feature in self.feature_names_:
            self._column = self.feature_names_.index(self.feature)
        else:
            self._column = None

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._column is None:
            return np.zeros(len(X), dtype="float64")
        return X[:, self._column].astype("float64")


class MajorityClassForecast(Model):
    """Predicts the majority class seen in training.

    The classification counterpart of the historical mean.  For up/down
    prediction on equities this is "always up", which beats 50% accuracy on most
    samples purely because of drift.  Any classifier that reports 52% accuracy
    should be compared against this number, not against a coin flip.
    """

    name = "majority_class"
    task: Task = "classification"
    min_train_samples = 5

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.majority_: float = 1.0
        self.rate_: float = 0.5

    def _fit(self, X, y, sample_weight) -> None:  # noqa: ARG002
        self.rate_ = float(np.mean(y))
        self.majority_ = 1.0 if self.rate_ >= 0.5 else 0.0

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_, dtype="float64")

    def predict_proba(self, X):
        return np.full(len(X), self.rate_, dtype="float64")


__all__ = [
    "HistoricalMeanForecast",
    "LastValueForecast",
    "MajorityClassForecast",
    "ZeroForecast",
]

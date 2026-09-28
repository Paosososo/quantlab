"""The common model interface.

Every model in this project implements the same four methods, so the
walk-forward runner, the experiment tracker and the strategy adapter never need
to know which one they are holding.  The interface is deliberately smaller than
scikit-learn's: ``fit``, ``predict``, ``get_params`` and ``clone``.

``clone`` matters more than it looks.  Walk-forward validation fits the same
configuration many times on different windows.  Reusing one fitted object across
folds would let state from fold 3 influence fold 4, which is a subtle form of
leakage that produces suspiciously good out-of-sample numbers.  Every fold gets
a fresh, unfitted instance built from the same parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantlab.exceptions import InsufficientDataError, NotFittedError

Task = Literal["regression", "classification"]


class Model(ABC):
    """Base class for every predictor."""

    #: Stable identifier used in ``model_runs.model_type`` and the registry.
    name: str = "abstract"
    task: Task = "regression"
    #: Minimum training rows before a fit is meaningful.
    min_train_samples: int = 30

    def __init__(self, **params: Any) -> None:
        self._params = dict(params)
        self._fitted = False
        self.feature_names_: list[str] = []

    # -- interface ---------------------------------------------------------
    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None: ...

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray: ...

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> Model:
        if len(X) != len(y):
            raise InsufficientDataError("X and y have different lengths", n_x=len(X), n_y=len(y))
        if len(X) < self.min_train_samples:
            raise InsufficientDataError(
                "not enough training rows",
                have=len(X),
                need=self.min_train_samples,
                model=self.name,
            )
        self.feature_names_ = list(X.columns)
        values = np.asarray(X, dtype="float64")
        targets = np.asarray(y, dtype="float64")
        if not np.isfinite(values).all():
            raise InsufficientDataError(
                "training features contain non-finite values", model=self.name
            )
        if not np.isfinite(targets).all():
            raise InsufficientDataError(
                "training targets contain non-finite values", model=self.name
            )
        self._fit(values, targets, sample_weight)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("predict() called before fit()", model=self.name)
        if self.feature_names_ and list(X.columns) != self.feature_names_:
            raise ValueError(
                f"{self.name}: feature columns changed between fit and predict; "
                f"expected {self.feature_names_[:5]}..., got {list(X.columns)[:5]}..."
            )
        return np.asarray(self._predict(np.asarray(X, dtype="float64")), dtype="float64")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:  # noqa: ARG002
        """Class-one probabilities for classifiers; None for regressors.

        The default ignores ``X`` because a regressor has no probabilities to
        report; classifiers override it.
        """
        return None

    # -- lifecycle ---------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def clone(self) -> Model:
        """A fresh, unfitted instance with identical parameters."""
        return type(self)(**self.get_params())

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "task": self.task, "params": self.get_params()}

    def feature_importance(self) -> pd.Series | None:
        """Per-feature importance where the model exposes one, else None."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self._params})"


__all__ = ["Model", "Task"]

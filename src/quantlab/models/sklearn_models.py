"""scikit-learn model wrappers.

Every estimator is wrapped in a ``Pipeline`` whose first step is a scaler, and
the whole pipeline is fitted inside each walk-forward fold.  That detail is the
point of the module: fitting a scaler on the full dataset and then splitting is
one of the most common leaks in applied ML, and it is invisible in the metrics
because the leak is small but systematic.  Putting the scaler in the pipeline
makes the correct behaviour the default rather than something to remember.

Model selection rationale (the research question is "does ML beat simple
baselines", so the ladder is deliberately short and interpretable):

* **Ridge** -- linear with L2 shrinkage.  With ~25 correlated features and a
  very low signal-to-noise ratio, unregularised OLS overfits badly; ridge is the
  minimum sensible linear model.
* **Lasso / ElasticNet** -- adds sparsity, which doubles as a feature-selection
  diagnostic.
* **Logistic regression** -- the classification counterpart.
* **Random forest** -- captures interactions and non-linearity without tuning,
  and its out-of-bag behaviour is well understood.
* **Histogram gradient boosting** -- usually the strongest tabular learner, and
  it ships with scikit-learn, so the base install needs no extra dependency.
  XGBoost is available behind an optional extra for comparison.

Deep learning is deliberately absent.  With a few thousand daily observations
and a signal-to-noise ratio near zero, a neural network's extra capacity buys
overfitting, not accuracy, and the honest baseline comparison matters more than
model complexity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quantlab.config import get_settings
from quantlab.models.base import Model, Task


class SklearnModel(Model):
    """Adapter from the project's :class:`Model` interface to scikit-learn."""

    # Annotated with the alias rather than inferred, so subclasses can narrow it
    # to "classification" without mypy treating that as an incompatible override.
    task: Task = "regression"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._pipeline: Pipeline | None = None

    def _build(self) -> Pipeline:  # pragma: no cover - overridden
        raise NotImplementedError

    def _fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        self._pipeline = self._build()
        target = y.astype("int64") if self.task == "classification" else y
        if sample_weight is not None:
            step = self._pipeline.steps[-1][0]
            self._pipeline.fit(X, target, **{f"{step}__sample_weight": sample_weight})
        else:
            self._pipeline.fit(X, target)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        assert self._pipeline is not None
        return self._pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray | None:
        if self.task != "classification" or self._pipeline is None:
            return None
        probabilities = self._pipeline.predict_proba(np.asarray(X, dtype="float64"))
        return np.asarray(probabilities[:, 1], dtype="float64")

    def feature_importance(self):
        import pandas as pd

        if self._pipeline is None or not self.feature_names_:
            return None
        estimator = self._pipeline.steps[-1][1]
        if hasattr(estimator, "coef_"):
            coefficients = np.ravel(estimator.coef_)
            if len(coefficients) == len(self.feature_names_):
                return pd.Series(coefficients, index=self.feature_names_).sort_values(
                    key=np.abs, ascending=False
                )
        if hasattr(estimator, "feature_importances_"):
            return pd.Series(estimator.feature_importances_, index=self.feature_names_).sort_values(
                ascending=False
            )
        return None

    def _seed(self) -> int:
        """Resolved random seed.

        ``random_state=None`` is a legitimate stored parameter (it means "use
        the project default"), so a plain ``dict.get`` with a fallback is not
        enough: the key exists and holds None.
        """
        value = self._params.get("random_state")
        return int(value) if value is not None else int(get_settings().random_seed)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------
class RidgeModel(SklearnModel):
    name = "ridge"
    task: Task = "regression"

    def __init__(self, alpha: float = 1.0, **params: Any) -> None:
        super().__init__(alpha=alpha, **params)

    def _build(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(self._params["alpha"]), random_state=None)),
            ]
        )


class LassoModel(SklearnModel):
    name = "lasso"
    task: Task = "regression"

    def __init__(self, alpha: float = 0.0001, max_iter: int = 5_000, **params: Any) -> None:
        super().__init__(alpha=alpha, max_iter=max_iter, **params)

    def _build(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    Lasso(
                        alpha=float(self._params["alpha"]),
                        max_iter=int(self._params["max_iter"]),
                    ),
                ),
            ]
        )


class ElasticNetModel(SklearnModel):
    name = "elastic_net"
    task: Task = "regression"

    def __init__(
        self, alpha: float = 0.0001, l1_ratio: float = 0.5, max_iter: int = 5_000, **params: Any
    ) -> None:
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, **params)

    def _build(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=float(self._params["alpha"]),
                        l1_ratio=float(self._params["l1_ratio"]),
                        max_iter=int(self._params["max_iter"]),
                    ),
                ),
            ]
        )


class LogisticModel(SklearnModel):
    name = "logistic"
    task: Task = "classification"

    def __init__(self, C: float = 1.0, max_iter: int = 2_000, **params: Any) -> None:
        super().__init__(C=C, max_iter=max_iter, **params)

    def _build(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(self._params["C"]),
                        max_iter=int(self._params["max_iter"]),
                        random_state=self._seed(),
                    ),
                ),
            ]
        )


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------
class RandomForestModel(SklearnModel):
    name = "random_forest"
    task: Task = "regression"
    min_train_samples = 100

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 6,
        min_samples_leaf: int = 20,
        max_features: float | str = 0.5,
        n_jobs: int = 1,
        random_state: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            **params,
        )

    def _estimator(self):
        # Shallow trees and large leaves on purpose.  Financial features carry
        # very little signal, so a deep forest memorises noise and its
        # out-of-sample R-squared goes negative.
        return RandomForestRegressor(
            n_estimators=int(self._params["n_estimators"]),
            max_depth=self._params["max_depth"],
            min_samples_leaf=int(self._params["min_samples_leaf"]),
            max_features=self._params["max_features"],
            n_jobs=int(self._params["n_jobs"]),
            random_state=self._seed(),
        )

    def _build(self) -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("model", self._estimator())])


class RandomForestClassifierModel(RandomForestModel):
    name = "random_forest_clf"
    task: Task = "classification"

    def _estimator(self):
        return RandomForestClassifier(
            n_estimators=int(self._params["n_estimators"]),
            max_depth=self._params["max_depth"],
            min_samples_leaf=int(self._params["min_samples_leaf"]),
            max_features=self._params["max_features"],
            n_jobs=int(self._params["n_jobs"]),
            random_state=self._seed(),
        )


class GradientBoostingModel(SklearnModel):
    name = "gradient_boosting"
    task: Task = "regression"
    min_train_samples = 100

    def __init__(
        self,
        learning_rate: float = 0.03,
        max_depth: int | None = 3,
        max_iter: int = 300,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        early_stopping: bool = False,
        random_state: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=early_stopping,
            random_state=random_state,
            **params,
        )

    def _estimator(self):
        # ``early_stopping`` defaults to False: scikit-learn's internal
        # validation split is random, which on time-series data would validate
        # on shuffled future observations.  Model selection here is the job of
        # the walk-forward runner, which splits chronologically.
        return HistGradientBoostingRegressor(
            learning_rate=float(self._params["learning_rate"]),
            max_depth=self._params["max_depth"],
            max_iter=int(self._params["max_iter"]),
            min_samples_leaf=int(self._params["min_samples_leaf"]),
            l2_regularization=float(self._params["l2_regularization"]),
            early_stopping=bool(self._params["early_stopping"]),
            random_state=self._seed(),
        )

    def _build(self) -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("model", self._estimator())])


class GradientBoostingClassifierModel(GradientBoostingModel):
    name = "gradient_boosting_clf"
    task: Task = "classification"

    def _estimator(self):
        return HistGradientBoostingClassifier(
            learning_rate=float(self._params["learning_rate"]),
            max_depth=self._params["max_depth"],
            max_iter=int(self._params["max_iter"]),
            min_samples_leaf=int(self._params["min_samples_leaf"]),
            l2_regularization=float(self._params["l2_regularization"]),
            early_stopping=bool(self._params["early_stopping"]),
            random_state=self._seed(),
        )


class XGBoostModel(SklearnModel):
    """Optional: only importable when the ``boost`` extra is installed."""

    name = "xgboost"
    task: Task = "regression"
    min_train_samples = 100

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        max_depth: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            **params,
        )

    def _build(self) -> Pipeline:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "XGBoost is an optional extra; install it with `pip install -e '.[boost]'`"
            ) from exc
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=int(self._params["n_estimators"]),
                        learning_rate=float(self._params["learning_rate"]),
                        max_depth=int(self._params["max_depth"]),
                        subsample=float(self._params["subsample"]),
                        colsample_bytree=float(self._params["colsample_bytree"]),
                        reg_lambda=float(self._params["reg_lambda"]),
                        random_state=self._seed(),
                        n_jobs=1,
                        verbosity=0,
                    ),
                ),
            ]
        )


__all__ = [
    "ElasticNetModel",
    "GradientBoostingClassifierModel",
    "GradientBoostingModel",
    "LassoModel",
    "LogisticModel",
    "RandomForestClassifierModel",
    "RandomForestModel",
    "RidgeModel",
    "SklearnModel",
    "XGBoostModel",
]

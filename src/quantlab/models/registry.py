"""Model registry: name to factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quantlab.exceptions import ModelError
from quantlab.models.base import Model
from quantlab.models.baselines import (
    HistoricalMeanForecast,
    LastValueForecast,
    MajorityClassForecast,
    ZeroForecast,
)
from quantlab.models.sklearn_models import (
    ElasticNetModel,
    GradientBoostingClassifierModel,
    GradientBoostingModel,
    LassoModel,
    LogisticModel,
    RandomForestClassifierModel,
    RandomForestModel,
    RidgeModel,
    XGBoostModel,
)

ModelFactory = Callable[..., Model]

_REGISTRY: dict[str, ModelFactory] = {
    "zero": ZeroForecast,
    "historical_mean": HistoricalMeanForecast,
    "last_value": LastValueForecast,
    "majority_class": MajorityClassForecast,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "elastic_net": ElasticNetModel,
    "logistic": LogisticModel,
    "random_forest": RandomForestModel,
    "random_forest_clf": RandomForestClassifierModel,
    "gradient_boosting": GradientBoostingModel,
    "gradient_boosting_clf": GradientBoostingClassifierModel,
    "xgboost": XGBoostModel,
}


def register_model(name: str, factory: ModelFactory) -> None:
    _REGISTRY[name] = factory


def get_model(name: str, **params: Any) -> Model:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise ModelError("unknown model", name=name, known=sorted(_REGISTRY)) from exc
    return factory(**params)


def available_models() -> list[str]:
    return sorted(_REGISTRY)


#: The default comparison ladder for the research question: two naive baselines,
#: one linear model, one bagged tree ensemble, one boosted one.
DEFAULT_REGRESSION_LADDER: list[tuple[str, dict[str, Any]]] = [
    ("zero", {}),
    ("historical_mean", {}),
    ("last_value", {"feature": "ret_1d"}),
    ("ridge", {"alpha": 10.0}),
    ("elastic_net", {"alpha": 0.0005, "l1_ratio": 0.5}),
    ("random_forest", {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 50}),
    ("gradient_boosting", {"learning_rate": 0.03, "max_depth": 3, "max_iter": 250}),
]


__all__ = [
    "DEFAULT_REGRESSION_LADDER",
    "ModelFactory",
    "available_models",
    "get_model",
    "register_model",
]

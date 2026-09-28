"""Model interface, baselines, estimators and time-series validation."""

from quantlab.models.base import Model
from quantlab.models.baselines import (
    HistoricalMeanForecast,
    LastValueForecast,
    MajorityClassForecast,
    ZeroForecast,
)
from quantlab.models.evaluation import ForecastMetrics, evaluate_forecasts
from quantlab.models.registry import DEFAULT_REGRESSION_LADDER, available_models, get_model
from quantlab.models.splitters import (
    ExpandingWindowSplitter,
    RollingWindowSplitter,
    SingleHoldoutSplitter,
    SplitterConfig,
    get_splitter,
)
from quantlab.models.tracking import ExperimentTracker
from quantlab.models.walkforward import WalkForwardResult, WalkForwardRunner, compare_models

__all__ = [
    "DEFAULT_REGRESSION_LADDER",
    "ExpandingWindowSplitter",
    "ExperimentTracker",
    "ForecastMetrics",
    "HistoricalMeanForecast",
    "LastValueForecast",
    "MajorityClassForecast",
    "Model",
    "RollingWindowSplitter",
    "SingleHoldoutSplitter",
    "SplitterConfig",
    "WalkForwardResult",
    "WalkForwardRunner",
    "ZeroForecast",
    "available_models",
    "compare_models",
    "evaluate_forecasts",
    "get_model",
    "get_splitter",
]

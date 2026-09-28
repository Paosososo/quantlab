"""Backtesting engine."""

from quantlab.backtesting.costs import (
    CostModel,
    FixedBpsSlippage,
    NoCommission,
    NoSlippage,
    PercentCommission,
    PerShareCommission,
    SquareRootImpactSlippage,
)
from quantlab.backtesting.engine import BacktestEngine, EngineConfig, EngineOutput
from quantlab.backtesting.execution import ExecutionModel
from quantlab.backtesting.market import MarketData
from quantlab.backtesting.metrics import PerformanceMetrics, compute_metrics
from quantlab.backtesting.portfolio import Portfolio
from quantlab.backtesting.results import BacktestResult, build_result, persist_result
from quantlab.backtesting.sizing import TargetWeightSizer, VolatilityTargetSizer
from quantlab.backtesting.strategy import (
    BuyAndHold,
    DecisionContext,
    MovingAverageCrossover,
    PredictionStrategy,
    Strategy,
    TimeSeriesMomentum,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BuyAndHold",
    "CostModel",
    "DecisionContext",
    "EngineConfig",
    "EngineOutput",
    "ExecutionModel",
    "FixedBpsSlippage",
    "MarketData",
    "MovingAverageCrossover",
    "NoCommission",
    "NoSlippage",
    "PerShareCommission",
    "PercentCommission",
    "PerformanceMetrics",
    "Portfolio",
    "PredictionStrategy",
    "SquareRootImpactSlippage",
    "Strategy",
    "TargetWeightSizer",
    "TimeSeriesMomentum",
    "VolatilityTargetSizer",
    "build_result",
    "compute_metrics",
    "persist_result",
]

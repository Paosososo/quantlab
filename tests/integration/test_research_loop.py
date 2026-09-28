"""The whole research loop, end to end, on data whose answer is known.

This is the most valuable test in the project.  It runs prices through feature
engineering, walk-forward model evaluation, prediction-driven backtesting and
statistical inference, twice:

* on a series with an **injected AR(1) component**, where a real signal exists;
* on a **pure random walk**, where none does.

The pipeline must find the first and must *not* find the second.  A system that
reports signal on a random walk is leaking.  A system that finds nothing in the
AR(1) series is broken.  Passing both directions is what makes a negative result
on real data believable.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from quantlab.backtesting import (
    BacktestEngine,
    EngineConfig,
    ExecutionModel,
    MarketData,
    PredictionStrategy,
    build_result,
)
from quantlab.backtesting.costs import ZERO_COST, CostModel, FixedBpsSlippage, PerShareCommission
from quantlab.backtesting.sizing import TargetWeightSizer
from quantlab.features import build_training_frame, default_feature_set
from quantlab.ingestion.providers.synthetic import SyntheticSpec, generate_price_frame
from quantlab.models import ExpandingWindowSplitter, SplitterConfig, WalkForwardRunner, get_model
from quantlab.research.hypothesis import diebold_mariano

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SYMBOLS = ("SYN_A", "SYN_B", "SYN_C")


def build_panel(autocorrelation: float) -> pd.DataFrame:
    spec = SyntheticSpec(
        start=dt.date(2006, 1, 1),
        end=dt.date(2020, 12, 31),
        autocorrelation=autocorrelation,
        regime_switching=False,
    )
    return pd.concat(
        [generate_price_frame(s, spec, seed=i) for i, s in enumerate(SYMBOLS)],
        ignore_index=True,
    )


def run_walk_forward(prices: pd.DataFrame, model_name: str, **params):
    feature_set = default_feature_set(include_macro=False)
    X, y, meta = build_training_frame(prices, feature_set, horizon=1)
    splitter = ExpandingWindowSplitter(
        SplitterConfig(n_splits=3, test_size=252, min_train_size=756, embargo=pd.Timedelta(days=2))
    )
    runner = WalkForwardRunner(splitter)
    return runner.run(get_model(model_name, **params), X, y, meta, seed=7)


@pytest.fixture(scope="module")
def predictable_result():
    return run_walk_forward(build_panel(0.25), "ridge", alpha=10.0)


@pytest.fixture(scope="module")
def random_walk_result():
    return run_walk_forward(build_panel(0.0), "ridge", alpha=10.0)


class TestSignalIsFoundWhenPresent:
    def test_information_coefficient_is_clearly_positive(self, predictable_result):
        assert predictable_result.overall.information_coefficient > 0.10

    def test_out_of_sample_r2_beats_the_zero_forecast(self, predictable_result):
        assert predictable_result.overall.r2_oos_vs_zero > 0.01

    def test_directional_accuracy_exceeds_a_coin_flip(self, predictable_result):
        assert predictable_result.overall.directional_accuracy > 0.53

    def test_every_fold_agrees(self, predictable_result):
        table = predictable_result.fold_metrics_frame()
        assert (table["information_coefficient"] > 0).all()


class TestNoSignalIsFoundWhenAbsent:
    def test_information_coefficient_is_near_zero(self, random_walk_result):
        """The leakage canary.  A leak would show up here as a large IC."""
        assert abs(random_walk_result.overall.information_coefficient) < 0.06

    def test_out_of_sample_r2_is_not_positive(self, random_walk_result):
        assert random_walk_result.overall.r2_oos_vs_zero < 0.01

    def test_directional_accuracy_is_near_half(self, random_walk_result):
        assert 0.46 < random_walk_result.overall.directional_accuracy < 0.54


class TestPredictionsAreTemporallyHonest:
    def test_every_prediction_carries_its_availability(self, predictable_result):
        assert "available_at" in predictable_result.predictions.columns
        assert predictable_result.predictions["available_at"].notna().all()

    def test_availability_is_never_before_the_decision_bar(self, predictable_result):
        frame = predictable_result.predictions
        assert (frame["available_at"] >= frame["ts"]).all()

    def test_folds_do_not_overlap_in_time(self, predictable_result):
        frame = predictable_result.predictions
        spans = frame.groupby("fold")["ts"].agg(["min", "max"]).sort_values("min")
        for (_, earlier), (_, later) in zip(
            spans.iterrows(), spans.iloc[1:].iterrows(), strict=False
        ):
            assert earlier["max"] < later["min"]


class TestPredictionDrivenBacktest:
    def test_a_real_signal_survives_zero_costs(self, predictable_result):
        prices = build_panel(0.25)
        market = MarketData(prices)
        engine = BacktestEngine(
            EngineConfig(
                initial_cash=1_000_000.0,
                execution=ExecutionModel(costs=ZERO_COST, max_participation=None),
                sizer=TargetWeightSizer(allow_fractional=True, max_participation=None),
                benchmark_symbol="SYN_A",
                max_borrow_fraction=1.0,
            )
        )
        output = engine.run(
            PredictionStrategy(list(SYMBOLS), long_threshold=0.0, short_threshold=-0.0),
            market,
            predictions=predictable_result.predictions,
        )
        result = build_result(output)
        assert result.metrics.n_trades > 0
        assert result.metrics.sharpe_ratio > 0.3

    def test_costs_materially_reduce_the_result(self, predictable_result):
        """The project's actual research question in miniature: does the edge
        survive realistic frictions?"""
        prices = build_panel(0.25)
        market = MarketData(prices)
        strategy_args = {"long_threshold": 0.0, "short_threshold": -0.0}

        def run(costs) -> float:
            engine = BacktestEngine(
                EngineConfig(
                    initial_cash=1_000_000.0,
                    execution=ExecutionModel(costs=costs, max_participation=None),
                    sizer=TargetWeightSizer(allow_fractional=True, max_participation=None),
                    max_borrow_fraction=1.0,
                )
            )
            output = engine.run(
                PredictionStrategy(list(SYMBOLS), **strategy_args),
                market,
                predictions=predictable_result.predictions,
            )
            return build_result(output).metrics.sharpe_ratio

        frictionless = run(ZERO_COST)
        realistic = run(CostModel(PerShareCommission(), FixedBpsSlippage(5.0)))
        assert realistic < frictionless

    def test_every_trade_respects_the_temporal_contract(self, predictable_result):
        prices = build_panel(0.25)
        engine = BacktestEngine(
            EngineConfig(
                initial_cash=1_000_000.0,
                execution=ExecutionModel(costs=ZERO_COST, max_participation=None),
                sizer=TargetWeightSizer(allow_fractional=True, max_participation=None),
                max_borrow_fraction=1.0,
            )
        )
        output = engine.run(
            PredictionStrategy(list(SYMBOLS), long_threshold=0.0),
            MarketData(prices),
            predictions=predictable_result.predictions,
        )
        assert output.fills
        assert all(f.execution_ts > f.decision_ts for f in output.fills)


class TestStatisticalComparison:
    def test_diebold_mariano_prefers_the_model_on_predictable_data(self, predictable_result):
        frame = predictable_result.predictions
        model_errors = (frame["y_true"] - frame["y_pred"]).to_numpy()
        zero_errors = frame["y_true"].to_numpy()
        outcome = diebold_mariano(model_errors, zero_errors)
        assert outcome.detail["favours"] == "a"
        assert outcome.rejects(0.05)

    def test_diebold_mariano_finds_no_difference_on_a_random_walk(self, random_walk_result):
        frame = random_walk_result.predictions
        model_errors = (frame["y_true"] - frame["y_pred"]).to_numpy()
        zero_errors = frame["y_true"].to_numpy()
        outcome = diebold_mariano(model_errors, zero_errors)
        assert not outcome.rejects(0.01)


class TestModelLadderOnRandomData:
    def test_no_model_in_the_ladder_finds_signal_in_noise(self):
        """Runs the whole comparison ladder on a random walk.

        Every model, including the flexible ones, must come out at roughly zero.
        A tree ensemble reporting a positive out-of-sample R-squared here would
        mean the folds are leaking.
        """
        from quantlab.models import compare_models

        prices = build_panel(0.0)
        X, y, meta = build_training_frame(
            prices, default_feature_set(include_macro=False), horizon=1
        )
        splitter = ExpandingWindowSplitter(
            SplitterConfig(n_splits=2, test_size=252, min_train_size=756)
        )
        ladder = [
            ("zero", {}),
            ("historical_mean", {}),
            ("ridge", {"alpha": 10.0}),
            ("random_forest", {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 50}),
        ]
        table, _ = compare_models(ladder, X, y, meta, splitter, seed=3)
        assert not table.empty
        assert (table["r2_oos_vs_zero"] < 0.02).all()
        finite_ic = table["information_coefficient"].dropna()
        assert (finite_ic.abs() < 0.08).all()

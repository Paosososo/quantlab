"""Model interface, baselines and forecast evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import InsufficientDataError, ModelError, NotFittedError
from quantlab.models.baselines import (
    HistoricalMeanForecast,
    LastValueForecast,
    MajorityClassForecast,
    ZeroForecast,
)
from quantlab.models.evaluation import (
    directional_accuracy,
    evaluate_forecasts,
    information_coefficient,
    out_of_sample_r2,
    roc_auc,
    root_mean_squared_error,
)
from quantlab.models.registry import available_models, get_model
from quantlab.models.sklearn_models import RidgeModel

pytestmark = pytest.mark.unit


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"ret_1d": x1, "x2": x2}), pd.Series(y, name="target")


class TestModelInterface:
    def test_predict_before_fit_raises(self, toy_data):
        X, _ = toy_data
        with pytest.raises(NotFittedError):
            RidgeModel().predict(X)

    def test_clone_returns_an_unfitted_copy(self, toy_data):
        X, y = toy_data
        model = RidgeModel(alpha=3.0).fit(X, y)
        clone = model.clone()
        assert clone.get_params() == model.get_params()
        assert not clone.is_fitted
        assert model.is_fitted

    def test_changing_the_feature_set_between_fit_and_predict_raises(self, toy_data):
        X, y = toy_data
        model = RidgeModel().fit(X, y)
        with pytest.raises(ValueError, match="feature columns changed"):
            model.predict(X.rename(columns={"x2": "x3"}))

    def test_too_few_rows_raises(self):
        model = RidgeModel()
        X = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(InsufficientDataError):
            model.fit(X, pd.Series([1.0, 2.0]))

    def test_non_finite_features_are_rejected(self, toy_data):
        X, y = toy_data
        broken = X.copy()
        broken.loc[0, "x2"] = np.inf
        with pytest.raises(InsufficientDataError, match="non-finite"):
            RidgeModel().fit(broken, y)

    def test_mismatched_lengths_are_rejected(self, toy_data):
        X, y = toy_data
        with pytest.raises(InsufficientDataError, match="different lengths"):
            RidgeModel().fit(X, y.iloc[:-1])

    def test_ridge_recovers_the_generating_coefficients(self, toy_data):
        """A sanity check that the pipeline wiring is right, not a claim about markets."""
        X, y = toy_data
        model = RidgeModel(alpha=0.01).fit(X, y)
        importance = model.feature_importance()
        assert importance is not None
        # Standardised coefficients keep the sign and relative magnitude.
        assert importance["ret_1d"] > 0
        assert importance["x2"] < 0
        assert abs(importance["ret_1d"]) > abs(importance["x2"])

    def test_the_scaler_lives_inside_the_model(self, toy_data):
        """Scaling must be part of the fitted object, or it leaks across folds."""
        X, y = toy_data
        model = RidgeModel().fit(X, y)
        assert model._pipeline is not None
        assert model._pipeline.steps[0][0] == "scaler"


class TestBaselines:
    def test_zero_forecast_predicts_zero(self, toy_data):
        X, y = toy_data
        model = ZeroForecast().fit(X, y)
        assert np.all(model.predict(X) == 0.0)

    def test_historical_mean_predicts_the_training_mean(self, toy_data):
        X, y = toy_data
        model = HistoricalMeanForecast().fit(X, y)
        assert model.predict(X)[0] == pytest.approx(y.mean())

    def test_historical_mean_ignores_the_test_distribution(self, toy_data):
        """The benchmark must come from training data only."""
        X, y = toy_data
        model = HistoricalMeanForecast().fit(X.iloc[:200], y.iloc[:200])
        assert model.predict(X.iloc[200:])[0] == pytest.approx(y.iloc[:200].mean())

    def test_last_value_echoes_the_named_feature(self, toy_data):
        X, y = toy_data
        model = LastValueForecast(feature="ret_1d").fit(X, y)
        np.testing.assert_allclose(model.predict(X), X["ret_1d"].to_numpy())

    def test_last_value_falls_back_to_zero_when_the_feature_is_absent(self, toy_data):
        X, y = toy_data
        model = LastValueForecast(feature="not_present").fit(X, y)
        assert np.all(model.predict(X) == 0.0)

    def test_majority_class_reports_the_training_rate(self):
        X = pd.DataFrame({"a": np.arange(100, dtype="float64")})
        y = pd.Series([1.0] * 60 + [0.0] * 40)
        model = MajorityClassForecast().fit(X, y)
        assert np.all(model.predict(X) == 1.0)
        assert model.predict_proba(X)[0] == pytest.approx(0.6)


class TestRegistry:
    def test_the_ladder_is_registered(self):
        assert {"zero", "historical_mean", "ridge", "random_forest", "gradient_boosting"} <= set(
            available_models()
        )

    def test_unknown_models_raise_with_a_helpful_list(self):
        with pytest.raises(ModelError, match="unknown model"):
            get_model("transformer")

    def test_parameters_reach_the_estimator(self):
        model = get_model("ridge", alpha=42.0)
        assert model.get_params()["alpha"] == 42.0


class TestForecastMetrics:
    def test_perfect_forecasts_score_perfectly(self):
        y = np.array([0.01, -0.02, 0.03, -0.01])
        assert root_mean_squared_error(y, y) == pytest.approx(0.0)
        assert out_of_sample_r2(y, y) == pytest.approx(1.0)
        assert directional_accuracy(y, y) == pytest.approx(1.0)

    def test_a_zero_forecast_scores_exactly_zero_r2(self):
        """By construction: the benchmark for R2_OS is the zero forecast."""
        y = np.array([0.01, -0.02, 0.03, -0.01])
        assert out_of_sample_r2(y, np.zeros_like(y)) == pytest.approx(0.0)

    def test_a_worse_than_useless_forecast_gives_negative_r2(self):
        y = np.array([0.01, -0.02, 0.03, -0.01])
        assert out_of_sample_r2(y, -y) < 0

    def test_directional_accuracy_ignores_zero_forecasts(self):
        y = np.array([0.01, -0.02, 0.03])
        predictions = np.array([0.0, -0.01, 0.02])
        assert directional_accuracy(y, predictions) == pytest.approx(1.0)

    def test_directional_accuracy_is_nan_for_an_all_zero_forecast(self):
        y = np.array([0.01, -0.02, 0.03])
        assert np.isnan(directional_accuracy(y, np.zeros_like(y)))

    def test_information_coefficient_is_a_correlation(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=500)
        noise = rng.normal(size=500)
        predictions = 0.5 * y + noise
        assert information_coefficient(y, predictions) == pytest.approx(
            np.corrcoef(y, predictions)[0, 1]
        )

    def test_roc_auc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(2)
        labels = rng.integers(0, 2, 300).astype("float64")
        scores = labels * 0.3 + rng.normal(size=300)
        assert roc_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-9)

    def test_roc_auc_is_nan_for_a_single_class(self):
        assert np.isnan(roc_auc(np.ones(50), np.random.default_rng(0).normal(size=50)))

    def test_evaluate_uses_the_supplied_training_mean(self):
        """A drifting series where the training mean is a good benchmark.

        Against the zero benchmark a zero forecast scores exactly 0; against a
        training mean that captures the drift it scores well below 0, which is
        the whole reason both numbers are reported.
        """
        rng = np.random.default_rng(3)
        y = 0.02 + rng.normal(scale=0.002, size=200)
        metrics = evaluate_forecasts(y, np.zeros_like(y), train_mean=0.02)
        assert metrics.r2_oos_vs_zero == pytest.approx(0.0)
        assert metrics.r2_oos_vs_mean < -50

    def test_r2_is_undefined_when_the_benchmark_is_perfect(self):
        """0/0 has no answer; returning a number here would be an invention."""
        y = np.full(50, 0.02)
        metrics = evaluate_forecasts(y, np.zeros_like(y), train_mean=0.02)
        assert np.isnan(metrics.r2_oos_vs_mean)

    def test_metrics_serialise_without_nan(self):
        payload = evaluate_forecasts(np.zeros(50), np.zeros(50)).to_dict()
        assert payload["information_coefficient"] is None
        assert isinstance(payload["n"], int)

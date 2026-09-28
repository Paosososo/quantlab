"""Risk metrics and portfolio optimisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import QuantlabError
from quantlab.risk.metrics import (
    conditional_value_at_risk,
    correlation_from_covariance,
    covariance_matrix,
    diversification_ratio,
    rolling_var,
    shrunk_covariance,
    summarise_risk,
    value_at_risk,
    var_breach_rate,
)
from quantlab.risk.optimization import (
    efficient_frontier,
    equal_weight,
    inverse_volatility,
    maximum_sharpe,
    minimum_variance,
    risk_contributions,
    risk_parity,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0004, 0.012, 2000))


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 1500
    market = rng.normal(0, 0.010, n)
    return pd.DataFrame(
        {
            "LOW": 0.4 * market + rng.normal(0, 0.004, n),
            "MID": 0.9 * market + rng.normal(0, 0.006, n),
            "HIGH": 1.5 * market + rng.normal(0, 0.012, n),
        }
    )


class TestValueAtRisk:
    def test_var_is_reported_as_a_loss(self, returns):
        assert value_at_risk(returns, 0.95) < 0

    def test_higher_confidence_means_a_larger_loss(self, returns):
        assert value_at_risk(returns, 0.99) < value_at_risk(returns, 0.95)

    def test_historical_var_is_the_empirical_quantile(self, returns):
        assert value_at_risk(returns, 0.95, "historical") == pytest.approx(
            float(np.quantile(returns.to_numpy(), 0.05))
        )

    def test_parametric_var_matches_the_formula(self, returns):
        from scipy import stats

        expected = returns.mean() + stats.norm.ppf(0.05) * returns.std(ddof=1)
        assert value_at_risk(returns, 0.95, "parametric") == pytest.approx(expected)

    def test_cornish_fisher_equals_parametric_on_normal_data(self, returns):
        """With near-zero skew and kurtosis the adjustment should almost vanish."""
        parametric = value_at_risk(returns, 0.95, "parametric")
        adjusted = value_at_risk(returns, 0.95, "cornish_fisher")
        assert adjusted == pytest.approx(parametric, abs=0.001)

    def test_cornish_fisher_is_more_severe_on_fat_tails(self):
        rng = np.random.default_rng(2)
        fat = pd.Series(rng.standard_t(df=3, size=5000) * 0.01)
        assert value_at_risk(fat, 0.99, "cornish_fisher") < value_at_risk(fat, 0.99, "parametric")

    def test_cvar_is_at_least_as_severe_as_var(self, returns):
        assert conditional_value_at_risk(returns, 0.95) <= value_at_risk(returns, 0.95)

    def test_an_unknown_method_raises(self, returns):
        with pytest.raises(ValueError, match="unknown VaR method"):
            value_at_risk(returns, 0.95, "magic")  # type: ignore[arg-type]

    def test_short_series_give_nan(self):
        assert np.isnan(value_at_risk(pd.Series([0.01, -0.01]), 0.95))

    def test_rolling_var_is_causal(self, returns):
        series = rolling_var(returns, window=252)
        assert series.iloc[:251].isna().all()
        assert not np.isnan(series.iloc[251])

    def test_breach_rate_is_near_the_nominal_level(self, returns):
        """Backtesting the risk model itself: a 95% VaR should break ~5% of days."""
        series = rolling_var(returns, window=500, confidence=0.95)
        rate = var_breach_rate(returns, series)
        assert 0.02 < rate < 0.09


class TestRiskSummary:
    def test_summary_is_internally_consistent(self, returns):
        summary = summarise_risk(returns)
        assert summary.n == len(returns)
        assert summary.cvar_95 <= summary.var_95_historical <= 0
        assert summary.max_drawdown <= 0
        assert summary.worst_day <= summary.best_day

    def test_summary_serialises_cleanly(self, returns):
        payload = summarise_risk(returns).to_dict()
        assert all(v is None or isinstance(v, int | float) for v in payload.values())

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="no finite returns"):
            summarise_risk(pd.Series([np.nan, np.nan]))


class TestCovariance:
    def test_annualisation_scales_by_periods(self, panel):
        daily = panel.cov()
        annual = covariance_matrix(panel)
        assert annual.iloc[0, 0] == pytest.approx(daily.iloc[0, 0] * 252)

    def test_shrinkage_pulls_off_diagonals_toward_zero(self, panel):
        sample = covariance_matrix(panel)
        shrunk = shrunk_covariance(panel, shrinkage=0.9)
        assert abs(shrunk.iloc[0, 1]) < abs(sample.iloc[0, 1])

    def test_ledoit_wolf_runs_without_a_manual_intensity(self, panel):
        shrunk = shrunk_covariance(panel)
        assert shrunk.shape == (3, 3)
        assert np.allclose(shrunk.to_numpy(), shrunk.to_numpy().T)

    def test_too_few_observations_raises(self):
        tiny = pd.DataFrame(np.random.default_rng(0).normal(size=(3, 5)))
        with pytest.raises(ValueError, match="more observations than assets"):
            shrunk_covariance(tiny)

    def test_correlation_has_a_unit_diagonal(self, panel):
        corr = correlation_from_covariance(covariance_matrix(panel))
        assert np.allclose(np.diag(corr.to_numpy()), 1.0)


class TestOptimisation:
    def test_equal_weight_is_uniform(self, panel):
        result = equal_weight(covariance_matrix(panel))
        assert result.weights.tolist() == pytest.approx([1 / 3] * 3)

    def test_weights_always_sum_to_one(self, panel):
        cov = shrunk_covariance(panel)
        for optimiser in (equal_weight, inverse_volatility, minimum_variance, risk_parity):
            assert optimiser(cov).weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_minimum_variance_beats_the_alternatives_in_sample(self, panel):
        """By construction: it minimises exactly this quantity on this matrix."""
        cov = shrunk_covariance(panel)
        minvar = minimum_variance(cov)
        assert minvar.converged
        for other in (equal_weight(cov), inverse_volatility(cov), risk_parity(cov)):
            assert minvar.volatility <= other.volatility + 1e-9

    def test_minimum_variance_favours_the_least_volatile_asset(self, panel):
        weights = minimum_variance(shrunk_covariance(panel)).weights
        assert weights["LOW"] > weights["HIGH"]

    def test_risk_parity_equalises_risk_contributions(self, panel):
        cov = shrunk_covariance(panel)
        result = risk_parity(cov)
        contributions = risk_contributions(result.weights, cov)
        assert contributions.max() - contributions.min() < 0.02

    def test_equal_weight_does_not_equalise_risk(self, panel):
        """The motivation for risk parity, demonstrated."""
        cov = shrunk_covariance(panel)
        contributions = risk_contributions(equal_weight(cov).weights, cov)
        assert contributions.max() - contributions.min() > 0.2

    def test_long_only_bounds_are_respected(self, panel):
        weights = minimum_variance(shrunk_covariance(panel)).weights
        assert (weights >= -1e-9).all()

    def test_shorting_can_be_enabled(self, panel):
        result = minimum_variance(shrunk_covariance(panel), allow_short=True, bounds=(-1.0, 1.0))
        assert result.converged

    def test_maximum_sharpe_concentrates_on_the_best_estimate(self, panel):
        """Documents the instability the docstring warns about."""
        cov = shrunk_covariance(panel)
        mu = panel.mean() * 252
        weights = maximum_sharpe(cov, mu).weights
        assert weights.max() > 0.4

    def test_diversification_ratio_exceeds_one_for_a_mixed_portfolio(self, panel):
        cov = shrunk_covariance(panel)
        assert diversification_ratio(equal_weight(cov).weights.to_numpy(), cov) > 1.0

    def test_the_frontier_is_upward_sloping(self, panel):
        frontier = efficient_frontier(shrunk_covariance(panel), panel.mean() * 252, n_points=8)
        assert not frontier.empty
        assert frontier["target_return"].is_monotonic_increasing

    def test_a_non_symmetric_covariance_is_rejected(self):
        bad = pd.DataFrame([[1.0, 0.5], [0.2, 1.0]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(QuantlabError, match="symmetric"):
            minimum_variance(bad)

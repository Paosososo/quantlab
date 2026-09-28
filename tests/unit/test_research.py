"""Statistical research utilities, checked against known-answer cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.research.descriptive import describe, drawdown_table, summary_table
from quantlab.research.hypothesis import (
    diebold_mariano,
    mean_return_t_test,
    sharpe_statistic,
    stationary_bootstrap_ci,
    stationary_bootstrap_indices,
)
from quantlab.research.multiple_testing import (
    adjust_pvalues,
    analyse_sharpe,
    benjamini_hochberg,
    bonferroni,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)
from quantlab.research.regression import newey_west_lags, ols, univariate_screen
from quantlab.research.stationarity import (
    adf_test,
    autocorrelation,
    classify_stationarity,
    kpss_test,
    volatility_clustering,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def white_noise() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0, 0.01, 1500))


@pytest.fixture
def random_walk(white_noise: pd.Series) -> pd.Series:
    return white_noise.cumsum() + 100.0


class TestDescriptive:
    def test_describe_matches_numpy(self, white_noise):
        stats_result = describe(white_noise)
        assert stats_result.n == len(white_noise)
        assert stats_result.mean == pytest.approx(white_noise.mean())
        assert stats_result.std == pytest.approx(white_noise.std(ddof=1))

    def test_describe_rejects_an_empty_series(self):
        with pytest.raises(ValueError, match="empty"):
            describe(pd.Series(dtype="float64"))

    def test_summary_table_skips_non_numeric_columns(self, white_noise):
        frame = pd.DataFrame({"a": white_noise, "label": ["x"] * len(white_noise)})
        table = summary_table(frame)
        assert list(table.index) == ["a"]

    def test_drawdown_table_finds_the_worst_episode(self):
        equity = pd.Series([100.0, 120.0, 60.0, 80.0, 130.0, 125.0])
        table = drawdown_table(equity, top=2)
        assert table.iloc[0]["depth"] == pytest.approx(-0.5)


class TestStationarity:
    def test_white_noise_is_stationary(self, white_noise):
        assert classify_stationarity(white_noise)["verdict"] == "stationary"

    def test_a_random_walk_has_a_unit_root(self, random_walk):
        assert classify_stationarity(random_walk)["verdict"] == "unit_root"

    def test_adf_and_kpss_have_opposite_nulls(self, white_noise):
        adf = adf_test(white_noise)
        kpss_result = kpss_test(white_noise)
        assert "unit root" in adf.null_hypothesis
        assert "stationary" in kpss_result.null_hypothesis
        assert adf.rejects(0.05)
        assert not kpss_result.rejects(0.05)

    def test_autocorrelation_bands_shrink_with_sample_size(self, white_noise):
        small = autocorrelation(white_noise.iloc[:200], nlags=10)
        large = autocorrelation(white_noise, nlags=10)
        assert large["upper_95"].iloc[0] < small["upper_95"].iloc[0]

    def test_autocorrelation_of_white_noise_is_within_the_bands(self, white_noise):
        table = autocorrelation(white_noise, nlags=15).iloc[1:]
        inside = table["acf"].between(table["lower_95"], table["upper_95"])
        assert inside.mean() > 0.8

    def test_volatility_clustering_is_detected_in_a_garch_like_series(self):
        rng = np.random.default_rng(1)
        n = 2000
        sigma = np.empty(n)
        sigma[0] = 0.01
        shocks = rng.normal(size=n)
        returns = np.empty(n)
        for i in range(1, n):
            sigma[i] = np.sqrt(1e-6 + 0.1 * returns[i - 1] ** 2 + 0.85 * sigma[i - 1] ** 2)
            returns[i] = sigma[i] * shocks[i]
        assert volatility_clustering(pd.Series(returns[1:]))["clustering_detected"]

    def test_no_clustering_in_iid_noise(self, white_noise):
        assert not volatility_clustering(white_noise)["clustering_detected"]


class TestRegression:
    def test_ols_recovers_known_coefficients(self):
        rng = np.random.default_rng(2)
        n = 2000
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 1.5 + 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.5, size=n)
        result = ols(pd.Series(y), pd.DataFrame({"x1": x1, "x2": x2}))
        assert result.params["x1"] == pytest.approx(2.0, abs=0.05)
        assert result.params["x2"] == pytest.approx(-0.5, abs=0.05)
        assert result.params["const"] == pytest.approx(1.5, abs=0.05)

    def test_hac_errors_are_larger_on_autocorrelated_residuals(self):
        """The reason HAC is the default: naive errors overstate significance."""
        rng = np.random.default_rng(3)
        n = 1000
        x = rng.normal(size=n)
        noise = pd.Series(rng.normal(size=n)).rolling(20, min_periods=1).mean().to_numpy()
        y = 0.1 * x + noise * 3
        naive = ols(pd.Series(y), pd.DataFrame({"x": x}), cov_type="nonrobust")
        robust = ols(pd.Series(y), pd.DataFrame({"x": x}), horizon=20)
        assert robust.std_errors["const"] > naive.std_errors["const"]

    def test_the_horizon_sets_the_hac_lag(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=500)
        y = 0.2 * x + rng.normal(size=500)
        result = ols(pd.Series(y), pd.DataFrame({"x": x}), horizon=21)
        assert result.hac_lags == 20

    def test_newey_west_bandwidth_grows_with_n(self):
        assert newey_west_lags(100) < newey_west_lags(10_000)

    def test_univariate_screen_ranks_by_absolute_t(self):
        rng = np.random.default_rng(5)
        n = 800
        strong = rng.normal(size=n)
        weak = rng.normal(size=n)
        y = 1.0 * strong + 0.02 * weak + rng.normal(size=n)
        table = univariate_screen(pd.Series(y), pd.DataFrame({"strong": strong, "weak": weak}))
        assert table.iloc[0]["feature"] == "strong"

    def test_no_overlap_raises(self):
        y = pd.Series([1.0, 2.0], index=[0, 1])
        X = pd.DataFrame({"x": [1.0, 2.0]}, index=[10, 11])
        with pytest.raises(ValueError, match="no overlapping"):
            ols(y, X)


class TestDieboldMariano:
    def test_a_clearly_better_forecast_is_detected(self):
        rng = np.random.default_rng(6)
        good = rng.normal(0, 1, 500)
        bad = rng.normal(0, 3, 500)
        result = diebold_mariano(good, bad)
        assert result.statistic < 0
        assert result.rejects(0.01)
        assert result.detail["favours"] == "a"

    def test_identical_forecasts_are_not_distinguishable(self):
        rng = np.random.default_rng(7)
        errors = rng.normal(0, 1, 500)
        result = diebold_mariano(errors, errors.copy())
        assert not result.rejects(0.05)

    def test_the_test_is_symmetric_under_swapping(self):
        rng = np.random.default_rng(8)
        a = rng.normal(0, 1, 400)
        b = rng.normal(0, 2, 400)
        forward = diebold_mariano(a, b)
        backward = diebold_mariano(b, a)
        assert forward.statistic == pytest.approx(-backward.statistic)

    def test_a_longer_horizon_widens_the_variance(self):
        """Overlapping forecasts carry less independent information."""
        rng = np.random.default_rng(9)
        a = rng.normal(0, 1, 600)
        b = rng.normal(0, 1.5, 600)
        one_step = diebold_mariano(a, b, horizon=1)
        many_step = diebold_mariano(a, b, horizon=21)
        assert abs(many_step.statistic) < abs(one_step.statistic)

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 10"):
            diebold_mariano(np.zeros(5), np.zeros(5))

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="same shape"):
            diebold_mariano(np.zeros(20), np.zeros(30))


class TestBootstrap:
    def test_indices_stay_in_range(self):
        rng = np.random.default_rng(10)
        indices = stationary_bootstrap_indices(500, 20.0, rng)
        assert indices.min() >= 0
        assert indices.max() < 500
        assert len(indices) == 500

    def test_the_bootstrap_preserves_dependence(self):
        """A block bootstrap should reproduce autocorrelation an iid resample destroys."""
        rng = np.random.default_rng(11)
        base = pd.Series(rng.normal(size=2000)).rolling(10, min_periods=1).mean().to_numpy()
        block_indices = stationary_bootstrap_indices(len(base), 50.0, rng)
        block_sample = base[block_indices]
        iid_sample = rng.permutation(base)
        block_rho = abs(np.corrcoef(block_sample[:-1], block_sample[1:])[0, 1])
        iid_rho = abs(np.corrcoef(iid_sample[:-1], iid_sample[1:])[0, 1])
        assert block_rho > iid_rho

    def test_a_sharpe_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(12)
        returns = rng.normal(0.0005, 0.01, 1000)
        interval = stationary_bootstrap_ci(returns, sharpe_statistic(252), n_boot=200, seed=1)
        assert interval["lower"] < interval["point_estimate"] < interval["upper"]

    def test_the_interval_is_wide_on_short_samples(self):
        """The corrective this project cares about: a three-year Sharpe is noisy."""
        rng = np.random.default_rng(13)
        returns = rng.normal(0.0006, 0.01, 756)
        interval = stationary_bootstrap_ci(returns, sharpe_statistic(252), n_boot=300, seed=2)
        assert interval["upper"] - interval["lower"] > 0.8

    def test_short_series_are_rejected(self):
        with pytest.raises(ValueError, match="at least 30"):
            stationary_bootstrap_ci(np.zeros(10), np.mean)

    def test_mean_return_t_test_finds_a_real_drift(self):
        rng = np.random.default_rng(14)
        returns = rng.normal(0.002, 0.005, 1000)
        assert mean_return_t_test(returns).rejects(0.01)


class TestMultipleTesting:
    def test_bonferroni_scales_by_the_number_of_tests(self):
        adjusted = bonferroni(np.array([0.01, 0.02, 0.03, 0.04]))
        assert adjusted[0] == pytest.approx(0.04)
        assert adjusted.max() <= 1.0

    def test_benjamini_hochberg_matches_statsmodels(self):
        from statsmodels.stats.multitest import multipletests

        pvalues = np.array([0.001, 0.008, 0.02, 0.04, 0.3, 0.6])
        expected = multipletests(pvalues, method="fdr_bh")[1]
        np.testing.assert_allclose(benjamini_hochberg(pvalues), expected, rtol=1e-12)

    def test_bh_is_less_conservative_than_bonferroni(self):
        pvalues = np.array([0.001, 0.008, 0.02, 0.04, 0.3, 0.6])
        assert (benjamini_hochberg(pvalues) <= bonferroni(pvalues) + 1e-12).all()

    def test_adjust_pvalues_returns_a_sorted_table(self):
        table = adjust_pvalues({"a": 0.3, "b": 0.001, "c": 0.05})
        assert list(table.index) == ["b", "c", "a"]

    def test_psr_rises_with_track_record_length(self):
        rng = np.random.default_rng(15)
        short = rng.normal(0.0005, 0.01, 200)
        long = rng.normal(0.0005, 0.01, 3000)
        assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)

    def test_psr_penalises_negative_skew(self):
        """Two series with an identical Sharpe; the left-tailed one scores worse.

        Both series are standardised and then given the same mean and standard
        deviation, so their Sharpe ratios agree to floating point and only the
        third and fourth moments differ.  Order matters here: rescaling after
        shifting would undo the mean match and the comparison would be against a
        different Sharpe, not a different shape.
        """
        rng = np.random.default_rng(16)
        symmetric = rng.normal(0.0005, 0.01, 4000)
        target_mean = float(symmetric.mean())
        target_std = float(symmetric.std(ddof=1))

        raw = -rng.gamma(2.0, 1.0, 4000)
        standardised = (raw - raw.mean()) / raw.std(ddof=1)
        skewed = standardised * target_std + target_mean

        sharpe_symmetric = symmetric.mean() / symmetric.std(ddof=1)
        sharpe_skewed = skewed.mean() / skewed.std(ddof=1)
        assert sharpe_skewed == pytest.approx(sharpe_symmetric, rel=1e-9)
        assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(symmetric)

    def test_expected_maximum_sharpe_grows_with_trials(self):
        assert expected_maximum_sharpe(100, 0.01) > expected_maximum_sharpe(10, 0.01)

    def test_a_single_trial_needs_no_deflation(self):
        assert expected_maximum_sharpe(1, 0.01) == 0.0

    def test_deflation_reduces_confidence(self):
        """The headline finding of the whole module."""
        rng = np.random.default_rng(17)
        returns = rng.normal(0.0006, 0.01, 1000)
        trials = rng.normal(0.0, 0.03, 100)
        undeflated = probabilistic_sharpe_ratio(returns)
        deflated = deflated_sharpe_ratio(returns, n_trials=100, trial_sharpes=trials)
        assert deflated < undeflated

    def test_deflation_needs_a_variance_estimate(self):
        with pytest.raises(ValueError, match="trial_sharpes or sharpe_variance"):
            deflated_sharpe_ratio(np.random.default_rng(0).normal(size=100), n_trials=10)

    def test_minimum_track_record_is_infinite_below_the_benchmark(self):
        rng = np.random.default_rng(18)
        losing = rng.normal(-0.001, 0.01, 500)
        assert np.isinf(minimum_track_record_length(losing))

    def test_analyse_sharpe_reports_everything(self):
        rng = np.random.default_rng(19)
        returns = rng.normal(0.0008, 0.01, 1260)
        result = analyse_sharpe(returns, n_trials=20, trial_sharpes=rng.normal(0, 0.02, 20))
        payload = result.to_dict()
        assert payload["n_observations"] == 1260
        assert 0.0 <= payload["probabilistic_sharpe"] <= 1.0
        assert payload["deflated_sharpe"] <= payload["probabilistic_sharpe"]
        assert payload["minimum_track_record_years"] > 0

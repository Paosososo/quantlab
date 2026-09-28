"""Performance metrics, checked against values computable by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.metrics import (
    annualised_return,
    annualised_turnover,
    annualised_volatility,
    beta_alpha,
    calmar_ratio,
    compute_metrics,
    drawdown_series,
    information_ratio,
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    to_returns,
    total_return,
    tracking_error,
    win_rate,
)

pytestmark = pytest.mark.unit


def curve(values: list[float]) -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=len(values), tz="UTC")
    return pd.Series(values, index=index, dtype="float64")


class TestReturnsAndLevels:
    def test_total_return_is_the_ratio_of_endpoints(self):
        assert total_return(curve([100.0, 150.0, 200.0])) == pytest.approx(1.0)

    def test_total_return_of_a_single_point_is_undefined(self):
        assert np.isnan(total_return(curve([100.0])))

    def test_to_returns_drops_the_first_observation(self):
        result = to_returns(curve([100.0, 110.0, 121.0]))
        assert len(result) == 2
        assert result.iloc[0] == pytest.approx(0.10)

    def test_annualised_return_is_geometric(self):
        """Doubling over one calendar year is roughly a 100% annualised return.

        Not exactly: 2020 was a leap year, so the span is 366 days against the
        365.25-day year used for scaling, which costs about 0.3 percentage
        points.  Using a fixed 365.25 rather than a per-year day count is the
        deliberate choice -- it keeps the metric comparable across periods
        instead of making leap years look better.
        """
        index = pd.DatetimeIndex(["2020-01-01", "2021-01-01"], tz="UTC")
        equity = pd.Series([100.0, 200.0], index=index)
        assert annualised_return(equity) == pytest.approx(1.0, rel=5e-3)

    def test_annualised_return_over_two_years_compounds(self):
        index = pd.DatetimeIndex(["2020-01-01", "2022-01-01"], tz="UTC")
        equity = pd.Series([100.0, 400.0], index=index)
        assert annualised_return(equity) == pytest.approx(1.0, rel=1e-2)


class TestRiskMetrics:
    def test_volatility_scales_by_the_square_root_of_periods(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.0, 0.01, 10_000))
        assert annualised_volatility(returns, 252) == pytest.approx(0.01 * np.sqrt(252), rel=0.05)

    def test_sharpe_of_a_constant_positive_return_is_infinite_or_nan(self):
        """Zero variance makes the ratio undefined; we return NaN, not a huge number."""
        returns = pd.Series(np.full(100, 0.001))
        assert np.isnan(sharpe_ratio(returns, 252))

    def test_sharpe_matches_the_definition(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.0005, 0.01, 2_000))
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        assert sharpe_ratio(returns, 252) == pytest.approx(expected)

    def test_a_risk_free_rate_lowers_the_sharpe(self):
        rng = np.random.default_rng(2)
        returns = pd.Series(rng.normal(0.0005, 0.01, 2_000))
        assert sharpe_ratio(returns, 252, 0.05) < sharpe_ratio(returns, 252, 0.0)

    def test_sortino_exceeds_sharpe_for_right_skewed_returns(self):
        """Upside volatility is not penalised, so a series with big gains and
        small losses scores better on Sortino than on Sharpe."""
        returns = pd.Series([-0.001] * 180 + [0.05] * 20)
        assert sortino_ratio(returns, 252) > sharpe_ratio(returns, 252)

    def test_sortino_with_no_downside_is_infinite(self):
        returns = pd.Series(np.full(50, 0.01))
        assert np.isinf(sortino_ratio(returns, 252))


class TestDrawdown:
    def test_drawdown_is_zero_on_a_monotone_rise(self):
        assert max_drawdown(curve([100.0, 110.0, 120.0])) == pytest.approx(0.0)

    def test_max_drawdown_is_peak_to_trough(self):
        assert max_drawdown(curve([100.0, 120.0, 60.0, 90.0])) == pytest.approx(-0.5)

    def test_drawdown_series_tracks_the_running_peak(self):
        dd = drawdown_series(curve([100.0, 120.0, 60.0, 120.0, 130.0]))
        assert dd.tolist() == pytest.approx([0.0, 0.0, -0.5, 0.0, 0.0])

    def test_drawdown_duration_counts_periods_under_water(self):
        assert max_drawdown_duration(curve([100.0, 90.0, 95.0, 99.0, 101.0])) == 3

    def test_calmar_is_annualised_return_over_max_drawdown(self):
        equity = curve([100.0, 120.0, 60.0, 150.0])
        expected = annualised_return(equity) / 0.5
        assert calmar_ratio(equity) == pytest.approx(expected)

    def test_calmar_is_undefined_without_a_drawdown(self):
        assert np.isnan(calmar_ratio(curve([100.0, 110.0, 120.0])))


class TestTradeStatistics:
    def test_win_rate_ignores_flat_periods(self):
        returns = pd.Series([0.01, -0.01, 0.0, 0.02, 0.0])
        assert win_rate(returns) == pytest.approx(2 / 3)

    def test_profit_factor_is_gross_gain_over_gross_loss(self):
        pnl = pd.Series([10.0, -5.0, 20.0, -5.0])
        assert profit_factor(pnl) == pytest.approx(3.0)

    def test_profit_factor_without_losses_is_infinite(self):
        assert np.isinf(profit_factor(pd.Series([1.0, 2.0])))

    def test_turnover_annualises_the_period_mean(self):
        turnover = pd.Series(np.full(252, 0.01))
        assert annualised_turnover(turnover, 252) == pytest.approx(2.52)


class TestBenchmarkRelative:
    def test_beta_of_a_series_against_itself_is_one(self):
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.normal(0, 0.01, 500))
        beta, alpha = beta_alpha(returns, returns, 252)
        assert beta == pytest.approx(1.0)
        assert alpha == pytest.approx(0.0, abs=1e-9)

    def test_double_beta_is_detected(self):
        rng = np.random.default_rng(4)
        benchmark = pd.Series(rng.normal(0, 0.01, 1000))
        strategy = 2.0 * benchmark
        beta, _ = beta_alpha(strategy, benchmark, 252)
        assert beta == pytest.approx(2.0)

    def test_tracking_error_is_zero_against_itself(self):
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0, 0.01, 300))
        assert tracking_error(returns, returns, 252) == pytest.approx(0.0)

    def test_information_ratio_is_undefined_against_itself(self):
        rng = np.random.default_rng(6)
        returns = pd.Series(rng.normal(0, 0.01, 300))
        assert np.isnan(information_ratio(returns, returns, 252))


class TestAggregate:
    def test_compute_metrics_populates_everything(self):
        rng = np.random.default_rng(7)
        path = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 756))
        equity = curve(list(path))
        bench = curve(list(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 756))))
        metrics = compute_metrics(
            equity,
            turnover=pd.Series(np.full(756, 0.02), index=equity.index),
            gross_exposure=equity * 0.9,
            benchmark_equity=bench,
            periods_per_year=252,
        )
        assert metrics.periods == 756
        assert np.isfinite(metrics.sharpe_ratio)
        assert metrics.max_drawdown <= 0.0
        assert np.isfinite(metrics.beta)
        assert metrics.average_gross_exposure == pytest.approx(0.9)

    def test_to_dict_replaces_non_finite_values_with_none(self):
        metrics = compute_metrics(curve([100.0, 100.0, 100.0]), periods_per_year=252)
        payload = metrics.to_dict()
        assert payload["sharpe_ratio"] is None
        assert isinstance(payload["periods"], int)

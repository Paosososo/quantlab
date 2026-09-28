"""Causal transform primitives: exact values on hand-checkable inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import LookAheadError
from quantlab.features import transforms as tf

pytestmark = pytest.mark.unit


@pytest.fixture
def series() -> pd.Series:
    return pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])


class TestBasics:
    def test_lag_shifts_forward_in_time(self, series):
        lagged = tf.lag(series, 1)
        assert np.isnan(lagged.iloc[0])
        assert lagged.iloc[1] == pytest.approx(100.0)

    def test_negative_lag_is_refused(self, series):
        with pytest.raises(LookAheadError, match="negative lag"):
            tf.lag(series, -1)

    def test_pct_change_is_exact(self, series):
        assert tf.pct_change(series, 1).iloc[1] == pytest.approx(0.10)

    def test_pct_change_does_not_fill_gaps(self):
        """A missing price should give a missing return, not a zero return."""
        gapped = pd.Series([100.0, np.nan, 121.0])
        result = tf.pct_change(gapped, 1)
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])

    def test_zero_lookback_is_refused(self, series):
        with pytest.raises(LookAheadError):
            tf.pct_change(series, 0)

    def test_log_return_matches_the_definition(self, series):
        assert tf.log_return(series, 1).iloc[1] == pytest.approx(np.log(110.0 / 100.0))

    def test_log_return_of_a_non_positive_price_is_nan(self):
        bad = pd.Series([100.0, -5.0, 120.0])
        result = tf.log_return(bad, 1)
        assert np.isnan(result.iloc[1])


class TestRollingWindows:
    def test_rolling_mean_requires_a_full_window(self, series):
        result = tf.rolling_mean(series, 3)
        assert result.iloc[:2].isna().all()
        assert result.iloc[2] == pytest.approx((100.0 + 110.0 + 121.0) / 3)

    def test_rolling_std_uses_the_sample_convention(self, series):
        result = tf.rolling_std(series, 3)
        expected = np.std([100.0, 110.0, 121.0], ddof=1)
        assert result.iloc[2] == pytest.approx(expected)

    def test_rolling_zscore_uses_only_the_trailing_window(self):
        values = pd.Series([1.0, 2.0, 3.0, 100.0, 5.0])
        result = tf.rolling_zscore(values, 3)
        window = np.array([1.0, 2.0, 3.0])
        expected = (3.0 - window.mean()) / window.std(ddof=1)
        assert result.iloc[2] == pytest.approx(expected)

    def test_rolling_zscore_is_unaffected_by_later_values(self):
        base = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        spiked = pd.Series([1.0, 2.0, 3.0, 4.0, 1e9])
        assert tf.rolling_zscore(base, 3).iloc[2] == pytest.approx(
            tf.rolling_zscore(spiked, 3).iloc[2]
        )

    def test_realised_volatility_annualises(self):
        returns = pd.Series(np.full(30, 0.01))
        result = tf.realised_volatility(returns, 20, periods_per_year=252)
        assert result.iloc[19] == pytest.approx(0.0, abs=1e-12)

    def test_downside_volatility_ignores_upside(self):
        symmetric = pd.Series([0.01, -0.01] * 20)
        only_up = pd.Series([0.01] * 40)
        assert tf.downside_volatility(symmetric, 20).iloc[-1] > 0
        assert tf.downside_volatility(only_up, 20).iloc[-1] == pytest.approx(0.0)

    def test_rolling_max_and_min(self, series):
        assert tf.rolling_max(series, 3).iloc[2] == pytest.approx(121.0)
        assert tf.rolling_min(series, 3).iloc[2] == pytest.approx(100.0)


class TestIndicators:
    def test_rsi_is_100_when_every_move_is_up(self):
        rising = pd.Series(np.arange(1, 60, dtype="float64"))
        assert tf.rsi(rising, 14).iloc[-1] == pytest.approx(100.0)

    def test_rsi_is_zero_when_every_move_is_down(self):
        falling = pd.Series(np.arange(60, 1, -1, dtype="float64"))
        assert tf.rsi(falling, 14).iloc[-1] == pytest.approx(0.0)

    def test_rsi_is_nan_on_a_flat_series(self):
        """0/0 carries no information; reporting a neutral 50 would invent one."""
        flat = pd.Series(np.full(60, 100.0))
        assert np.isnan(tf.rsi(flat, 14).iloc[-1])

    def test_rsi_stays_within_bounds(self):
        rng = np.random.default_rng(0)
        noisy = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
        values = tf.rsi(noisy, 14).dropna()
        assert values.between(0.0, 100.0).all()

    def test_atr_uses_the_previous_close(self):
        high = pd.Series([11.0, 12.0, 13.0] * 10)
        low = pd.Series([9.0, 10.0, 11.0] * 10)
        close = pd.Series([10.0, 11.0, 12.0] * 10)
        atr = tf.average_true_range(high, low, close, 14)
        assert atr.dropna().gt(0).all()

    def test_ewm_requires_a_full_span_before_emitting(self):
        values = pd.Series(np.arange(50, dtype="float64"))
        result = tf.ewm_mean(values, 20)
        assert result.iloc[:19].isna().all()
        assert not np.isnan(result.iloc[19])


class TestForwardReturnIsALabel:
    def test_forward_return_looks_ahead_by_design(self, series):
        result = tf.forward_return(series, 1)
        assert result.iloc[0] == pytest.approx(0.10)
        assert np.isnan(result.iloc[-1])

    def test_forward_return_rejects_a_zero_horizon(self, series):
        with pytest.raises(ValueError, match="positive"):
            tf.forward_return(series, 0)

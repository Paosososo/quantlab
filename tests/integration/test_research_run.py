"""The research study module.

Runs the whole study offline at reduced size.  What is being checked is not the
numbers -- those depend on the data -- but that the study reports whatever it
finds, in both directions, and that its verdict logic distinguishes a baseline
from a machine-learning model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.research.hypothesis import diebold_mariano_loss_differential
from quantlab.research_run import (
    BASELINE_MODELS,
    COST_SCENARIOS,
    _build_verdict,
    _daily_panel_loss_difference,
    load_price_panel,
    render_report,
    run_research_study,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> dict:
    """One reduced-size offline study, reused across the module."""
    return run_research_study(
        symbols=["SYN_BROAD", "SYN_TREND"],
        horizon=1,
        use_database=False,
        n_splits=2,
        test_size=126,
        min_train_size=504,
        output_dir=tmp_path_factory.mktemp("study"),
    )


class TestDataLoading:
    def test_the_offline_panel_is_generated_deterministically(self):
        first, source, symbols = load_price_panel(["SYN_BROAD"], use_database=False)
        second, _, _ = load_price_panel(["SYN_BROAD"], use_database=False)
        assert source == "synthetic"
        assert symbols == ["SYN_BROAD"]
        pd.testing.assert_frame_equal(first, second)

    def test_the_panel_carries_availability(self):
        panel, _, _ = load_price_panel(["SYN_BROAD"], use_database=False)
        assert (panel["available_at"] > panel["ts"]).all()


class TestCostScenarios:
    def test_three_scenarios_of_increasing_severity(self):
        assert list(COST_SCENARIOS) == ["frictionless", "realistic", "pessimistic"]

    def test_frictionless_really_is_free(self):
        model = COST_SCENARIOS["frictionless"]
        assert model.commission.charge(1_000, 100.0) == 0.0

    def test_pessimistic_costs_more_than_realistic(self):
        realistic = COST_SCENARIOS["realistic"].commission.charge(1_000, 100.0)
        pessimistic = COST_SCENARIOS["pessimistic"].commission.charge(1_000, 100.0)
        assert pessimistic > realistic


class TestStudyOutput:
    def test_the_study_writes_both_artefacts(self, study):
        assert Path(study["report"]).exists()
        assert Path(study["results"]).exists()

    def test_the_json_carries_every_section(self, study):
        payload = json.loads(Path(study["results"]).read_text())
        assert {
            "forecast_accuracy",
            "statistical_significance",
            "economic_significance",
            "data_snooping",
            "stationarity",
            "verdict",
        } <= set(payload)

    def test_every_model_appears_in_the_output(self, study):
        payload = json.loads(Path(study["results"]).read_text())
        models = {row["model"] for row in payload["forecast_accuracy"]}
        # No step selects a subset to report; the whole ladder is always shown.
        assert "zero" in models
        assert len(models) >= 5

    def test_the_zero_baseline_scores_exactly_zero(self, study):
        payload = json.loads(Path(study["results"]).read_text())
        zero = next(r for r in payload["forecast_accuracy"] if r["model"] == "zero")
        assert zero["r2_oos_vs_zero"] == pytest.approx(0.0)

    def test_all_three_cost_scenarios_are_evaluated(self, study):
        payload = json.loads(Path(study["results"]).read_text())
        scenarios = {row["scenario"] for row in payload["economic_significance"]}
        assert scenarios == {"frictionless", "realistic", "pessimistic"}

    def test_reference_strategies_are_included_for_comparison(self, study):
        payload = json.loads(Path(study["results"]).read_text())
        references = {
            row["strategy"]
            for row in payload["economic_significance"]
            if row["kind"] == "reference"
        }
        assert {"buy_and_hold", "ma_crossover", "ts_momentum"} <= references

    def test_prices_have_a_unit_root_and_returns_do_not(self, study):
        payload = json.loads(Path(study["results"]).read_text())
        assert payload["stationarity"]["prices"] == "unit_root"
        assert payload["stationarity"]["returns"] == "stationary"

    def test_the_report_states_it_used_synthetic_data(self, study):
        report = Path(study["report"]).read_text()
        assert "synthetic generator, not market data" in report

    def test_the_report_contains_every_section(self, study):
        report = Path(study["report"]).read_text()
        for heading in (
            "## Verdict",
            "## 1. Forecast accuracy",
            "## 2. Statistical significance",
            "## 3. Economic significance",
            "## 4. Data-snooping correction",
            "## How to read this critically",
        ):
            assert heading in report

    def test_the_report_warns_about_implausible_results(self, study):
        report = Path(study["report"]).read_text()
        assert "treated as a bug until proven otherwise" in report

    def test_significance_uses_dates_as_sample_units(self, study):
        payload = json.loads(Path(study["results"]).read_text())
        assert "n counts distinct dates" in payload["significance_method"]
        for row in payload["statistical_significance"]:
            assert row["n_asset_date_pairs"] >= row["n"] >= 30
            assert row["hac_lags"] >= 1


class TestPanelInference:
    @staticmethod
    def _joined(symbols: list[str]) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=60, freq="B", tz="UTC")
        index = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "ts"])
        signal = np.tile(np.sin(np.arange(len(dates))), len(symbols))
        return pd.DataFrame(
            {
                "y_true": signal,
                "y_true_base": signal,
                "y_pred": signal + 0.3,
                "y_pred_base": signal + 0.5,
            },
            index=index,
        )

    def test_repeating_identical_assets_does_not_inflate_significance(self):
        one = _daily_panel_loss_difference(self._joined(["AAA"]))
        three = _daily_panel_loss_difference(self._joined(["AAA", "BBB", "CCC"]))
        pd.testing.assert_series_equal(one, three, check_freq=False)
        first = diebold_mariano_loss_differential(one)
        repeated = diebold_mariano_loss_differential(three)
        assert first.pvalue == pytest.approx(repeated.pvalue)
        assert repeated.detail["n"] == 60

    def test_asset_order_does_not_change_daily_loss(self):
        joined = self._joined(["AAA", "BBB"])
        expected = _daily_panel_loss_difference(joined)
        actual = _daily_panel_loss_difference(joined.iloc[::-1])
        pd.testing.assert_series_equal(actual, expected)

    def test_mismatched_targets_are_rejected(self):
        joined = self._joined(["AAA"])
        joined.iloc[0, joined.columns.get_loc("y_true_base")] = 99.0
        with pytest.raises(ValueError, match="targets differ"):
            _daily_panel_loss_difference(joined)


class TestVerdictLogic:
    """The verdict must be able to say no, and must not confuse a baseline with ML."""

    @staticmethod
    def _tables(
        r2: dict[str, float],
        favours: dict[str, str],
        p_values: dict[str, float],
    ):
        """Build the three tables `_build_verdict` consumes.

        `beats_baseline_at_5pct` is derived the way the real pipeline derives it:
        significant *and* favouring the model.  A model with a small p-value that
        favours the baseline is significantly worse, not better.
        """
        forecast = pd.DataFrame(
            {"r2_oos_vs_zero": list(r2.values())}, index=pd.Index(list(r2), name="model")
        )
        significance = pd.DataFrame(
            {
                "model": list(favours),
                "favours": list(favours.values()),
                "adjusted_p_value": [p_values[m] for m in favours],
                "beats_baseline_at_5pct": [
                    p_values[m] < 0.05 and favours[m] == "a" for m in favours
                ],
            }
        )
        economic = pd.DataFrame(
            [
                {
                    "scenario": "frictionless",
                    "kind": "model",
                    "strategy": "ridge",
                    "sharpe_ratio": 1.0,
                },
                {
                    "scenario": "realistic",
                    "kind": "model",
                    "strategy": "ridge",
                    "sharpe_ratio": 0.4,
                },
                {
                    "scenario": "realistic",
                    "kind": "reference",
                    "strategy": "buy_and_hold",
                    "sharpe_ratio": 0.9,
                },
            ]
        )
        return forecast, significance, economic

    def test_no_positive_r2_gives_a_clean_negative(self):
        forecast, significance, economic = self._tables(
            {"zero": 0.0, "ridge": -0.01, "random_forest": -0.02},
            {"ridge": "b", "random_forest": "b"},
            {"ridge": 0.001, "random_forest": 0.004},
        )
        verdict, notes = _build_verdict(forecast, significance, economic, "synthetic", {})
        assert "the answer to the research question is no" in verdict
        assert any("significantly *worse*" in n for n in notes)

    def test_a_significant_baseline_does_not_count_as_evidence_for_ml(self):
        """The category error the code is written to avoid."""
        forecast, significance, economic = self._tables(
            {"zero": 0.0, "historical_mean": 0.002, "ridge": -0.01},
            {"historical_mean": "a", "ridge": "b"},
            {"historical_mean": 0.001, "ridge": 0.9},
        )
        verdict, notes = _build_verdict(forecast, significance, economic, "synthetic", {})
        assert "historical_mean" not in verdict
        assert any("not evidence for machine learning" in n for n in notes)

    def test_statistically_real_but_economically_irrelevant_is_reported_as_such(self):
        forecast, significance, economic = self._tables(
            {"zero": 0.0, "ridge": 0.01},
            {"ridge": "a"},
            {"ridge": 0.001},
        )
        verdict, _ = _build_verdict(forecast, significance, economic, "synthetic", {})
        assert "economically irrelevant" in verdict

    def test_a_win_is_reported_with_a_caution(self):
        forecast, significance, economic = self._tables(
            {"zero": 0.0, "ridge": 0.02},
            {"ridge": "a"},
            {"ridge": 0.001},
        )
        economic.loc[economic["strategy"] == "ridge", "sharpe_ratio"] = 2.0
        verdict, _ = _build_verdict(forecast, significance, economic, "database", {})
        assert "Treat this as provisional" in verdict

    def test_synthetic_data_is_always_flagged(self):
        forecast, significance, economic = self._tables(
            {"zero": 0.0, "ridge": 0.01}, {"ridge": "a"}, {"ridge": 0.001}
        )
        _, notes = _build_verdict(forecast, significance, economic, "synthetic", {})
        assert any("synthetic generator" in n for n in notes)

    def test_baselines_are_named_explicitly(self):
        assert {"zero", "historical_mean", "last_value"} <= BASELINE_MODELS


class TestReportRendering:
    def test_an_empty_significance_table_renders_without_crashing(self, study):
        import datetime as dt

        from quantlab.research_run import StudyResult

        payload = json.loads(Path(study["results"]).read_text())
        empty = StudyResult(
            generated_at=dt.datetime.now(tz=dt.timezone.utc),
            data_source="synthetic",
            symbols=["A"],
            horizon=1,
            n_observations=0,
            n_features=0,
            feature_set_hash="x" * 64,
            sample_start="2020-01-01",
            sample_end="2020-12-31",
            evaluation_start="2020-07-01",
            evaluation_end="2020-12-31",
            forecast_table=pd.DataFrame(),
            significance_table=pd.DataFrame(),
            economic_table=pd.DataFrame(),
            stationarity=payload["stationarity"],
            snooping={},
            verdict="nothing to report",
        )
        report = render_report(empty)
        assert "_No results._" in report

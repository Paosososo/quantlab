"""Feature generation.

Recomputes the whole feature set rather than appending the latest day.  The
reason is reproducibility: a feature set is identified by the hash of its
definition, and a table built partly under one definition and partly under
another has no single hash and cannot be reproduced.  Recomputation is cheap at
this data volume, and the upsert makes it idempotent.
"""

from __future__ import annotations

from typing import Any

from _common import dag_defaults, summarise
from airflow.decorators import dag, task


@dag(
    dag_id="feature_generation",
    description="Compute causal features and persist them",
    schedule="0 1 * * 2-6",  # after the previous evening's ingestion, in UTC
    **dag_defaults(),
)
def feature_generation() -> None:
    @task
    def check_upstream() -> dict[str, Any]:
        """Refuse to build features on stale prices.

        This DAG is scheduled by cron two hours after ``market_data_ingestion``,
        not chained to it.  Cron cannot tell whether that run succeeded, so
        without this gate a failed ingestion produces a *successful* feature run
        over last week's data -- the worst kind of failure, because nothing goes
        red.  See ``assert_upstream_fresh`` for why this checks the data rather
        than using a cross-DAG sensor.
        """
        from quantlab.pipelines import assert_upstream_fresh

        return assert_upstream_fresh(context="feature_generation")

    @task
    def build_features(_gate: dict[str, Any]) -> dict[str, Any]:
        from quantlab.pipelines import run_feature_generation

        return summarise(run_feature_generation(include_macro=True))

    @task
    def check_leakage(result: dict[str, Any]) -> dict[str, Any]:
        """Re-run the leakage harness against the live feature definitions.

        The unit tests already do this on synthetic data.  Repeating it in the
        DAG catches the case that matters operationally: someone edits a feature,
        the change passes review, and the pipeline starts producing a leaky
        column in production.  It runs on generated data, so it needs no network
        and adds seconds, not minutes.
        """
        import datetime as dt
        import functools

        from quantlab.features.guards import assert_causal
        from quantlab.features.library import default_feature_set
        from quantlab.features.pipeline import compute_symbol_features
        from quantlab.ingestion.providers.synthetic import SyntheticSpec, generate_price_frame
        from quantlab.logging import get_logger

        log = get_logger("dags.feature_generation")
        feature_set = default_feature_set(include_macro=False)
        probe = generate_price_frame(
            "LEAK_PROBE",
            SyntheticSpec(start=dt.date(2015, 1, 1), end=dt.date(2022, 12, 31)),
            seed=1,
        )
        report = assert_causal(
            functools.partial(compute_symbol_features, feature_set=feature_set),
            probe,
            input_columns=["open", "high", "low", "close", "adj_close", "volume"],
        )
        log.info("features.leakage_check_passed", features=len(report.checked_features))
        return {"clean": report.clean, "n_features": len(report.checked_features), **result}

    check_leakage(build_features(check_upstream()))


feature_generation()

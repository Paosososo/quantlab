"""Post-load data-quality checks.

Runs after ingestion and writes every result to ``data_quality_checks``, so data
quality is a queryable time series rather than a line in a scheduler log.

The task fails the DAG on a FAILED check and passes on a WARNED one.  That split
is deliberate: real market data always contains anomalies (a 15% single-day
move, a holiday gap), and a pipeline that halts on every anomaly is a pipeline
that gets switched off.  Failures are reserved for conditions that make the data
unusable, such as a symbol with no rows at all.
"""

from __future__ import annotations

from typing import Any

from _common import dag_defaults, summarise
from airflow.decorators import dag, task
from airflow.sensors.external_task import ExternalTaskSensor


@dag(
    dag_id="data_validation",
    description="Validate the warehouse after ingestion",
    schedule="30 23 * * 1-5",
    **dag_defaults(),
)
def data_validation() -> None:
    wait_for_ingestion = ExternalTaskSensor(
        task_id="wait_for_ingestion",
        external_dag_id="market_data_ingestion",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        mode="reschedule",  # frees a worker slot while waiting
        poke_interval=300,
        timeout=60 * 60 * 2,
        execution_delta=__import__("datetime").timedelta(minutes=30),
    )

    @task
    def validate_warehouse() -> dict[str, Any]:
        from quantlab.pipelines import run_data_validation

        return summarise(run_data_validation(fail_on_error=True))

    @task
    def report_issues(result: dict[str, Any]) -> dict[str, Any]:
        """Surface warnings in the task log where an operator will see them."""
        from quantlab.logging import get_logger

        log = get_logger("dags.data_validation")
        issues = result.get("issues", []) if isinstance(result, dict) else []
        for issue in issues:
            log.warning("validation.issue", **issue)
        return {"n_issues": len(issues)}

    wait_for_ingestion >> report_issues(validate_warehouse())


data_validation()

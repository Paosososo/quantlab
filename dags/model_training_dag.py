"""Weekly walk-forward model training.

Weekly, not daily.  Refitting a model on one extra day of data changes almost
nothing, and running it daily multiplies the number of configurations that have
been evaluated -- which is exactly the multiple-testing problem the research
layer exists to correct for.  A slower cadence keeps the trial count honest.

Every run writes a manifest and a ``model_runs`` row carrying the config hash,
the data fingerprint and the seed, so any reported number can be traced to the
configuration that produced it.
"""

from __future__ import annotations

from typing import Any

from _common import dag_defaults, summarise
from airflow.decorators import dag, task


@dag(
    dag_id="model_training",
    description="Walk-forward training and evaluation of the model ladder",
    schedule="0 3 * * 6",  # Saturday, once the week's data is complete
    **dag_defaults(execution_timeout_minutes=120),
)
def model_training() -> None:
    @task
    def check_upstream() -> dict[str, Any]:
        """Refuse to retrain on stale prices.

        Same reasoning as in ``feature_generation``: this DAG's Saturday cron is
        not chained to the weekday ingestion runs, so a silent ingestion outage
        would otherwise produce a model trained on an old panel and a
        ``model_runs`` row that looks perfectly healthy.
        """
        from quantlab.pipelines import assert_upstream_fresh

        return assert_upstream_fresh(context="model_training")

    @task(execution_timeout=__import__("datetime").timedelta(hours=2))
    def train_ladder(_gate: dict[str, Any]) -> dict[str, Any]:
        from quantlab.pipelines import run_model_training

        return summarise(run_model_training(run_prefix="airflow"))

    @task
    def compare_to_baseline(result: dict[str, Any]) -> dict[str, Any]:
        """Log whether any model beat the naive baseline this week.

        Deliberately reports the comparison rather than acting on it.  An
        automated "promote the best model" step would optimise the pipeline
        against its own walk-forward score, which is overfitting with extra
        infrastructure.
        """
        from quantlab.logging import get_logger

        log = get_logger("dags.model_training")
        results = result.get("results", []) if isinstance(result, dict) else []
        baseline = next((r for r in results if r.get("model") == "zero" and r.get("metrics")), None)
        if baseline is None:
            log.warning("model_training.no_baseline")
            return {"compared": False}

        baseline_rmse = baseline["metrics"].get("rmse")
        beat = [
            r["model"]
            for r in results
            if r.get("metrics")
            and r["model"] != "zero"
            and baseline_rmse is not None
            and (r["metrics"].get("rmse") or float("inf")) < baseline_rmse
        ]
        log.info(
            "model_training.comparison",
            baseline_rmse=baseline_rmse,
            models_beating_baseline=beat,
            n_models=len(results),
        )
        return {"compared": True, "baseline_rmse": baseline_rmse, "beat_baseline": beat}

    compare_to_baseline(train_ladder(check_upstream()))


model_training()

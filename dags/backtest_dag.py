"""Weekly backtesting of the classical strategy set.

Runs after model training so that a prediction-driven strategy can use the
week's forecasts.  Results land in ``backtests``, ``orders``, ``trades`` and
``portfolio_snapshots``, which is what the API and the dashboard read.

Each run also recomputes the deflated Sharpe ratio across the strategies tested,
so the reported numbers carry their own multiple-testing correction rather than
being presented as if only one strategy had ever been tried.
"""

from __future__ import annotations

from typing import Any

from _common import dag_defaults, summarise
from airflow.decorators import dag, task
from airflow.sensors.external_task import ExternalTaskSensor


@dag(
    dag_id="backtesting",
    description="Run and persist strategy backtests, with data-snooping correction",
    schedule="0 6 * * 6",
    **dag_defaults(),
)
def backtesting() -> None:
    wait_for_training = ExternalTaskSensor(
        task_id="wait_for_model_training",
        external_dag_id="model_training",
        allowed_states=["success"],
        failed_states=["failed", "skipped"],
        mode="reschedule",
        poke_interval=600,
        timeout=60 * 60 * 4,
        execution_delta=__import__("datetime").timedelta(hours=3),
    )

    @task
    def run_strategies() -> dict[str, Any]:
        from quantlab.pipelines import run_backtests

        return summarise(run_backtests())

    @task
    def deflate_sharpe(result: dict[str, Any]) -> dict[str, Any]:
        """Correct the reported Sharpe ratios for the number of strategies tried.

        The naive alternative -- reporting the best strategy's Sharpe as though
        it were the only one tested -- is the single most common way backtest
        results mislead.  See :mod:`quantlab.research.multiple_testing`.
        """
        import numpy as np

        from quantlab.logging import get_logger
        from quantlab.research.multiple_testing import expected_maximum_sharpe

        log = get_logger("dags.backtesting")
        results = result.get("results", []) if isinstance(result, dict) else []
        annual = [
            r["metrics"]["sharpe_ratio"]
            for r in results
            if r.get("metrics") and r["metrics"].get("sharpe_ratio") is not None
        ]
        if len(annual) < 2:
            log.warning("backtesting.too_few_strategies_to_deflate", n=len(annual))
            return {"deflated": False, "n_strategies": len(annual)}

        per_period = np.asarray(annual, dtype="float64") / np.sqrt(252.0)
        threshold = expected_maximum_sharpe(len(per_period), float(np.var(per_period, ddof=1)))
        best = float(np.max(per_period))
        log.info(
            "backtesting.deflation",
            n_strategies=len(annual),
            best_sharpe_annual=round(best * np.sqrt(252.0), 4),
            selection_threshold_annual=round(threshold * np.sqrt(252.0), 4),
            clears_threshold=bool(best > threshold),
        )
        return {
            "deflated": True,
            "n_strategies": len(annual),
            "best_sharpe_annual": best * float(np.sqrt(252.0)),
            "selection_threshold_annual": threshold * float(np.sqrt(252.0)),
            "clears_threshold": bool(best > threshold),
        }

    # ``wait_for_training >> deflate_sharpe(run_strategies())`` was wrong.  The
    # TaskFlow call already makes ``run_strategies`` upstream of
    # ``deflate_sharpe``; chaining the sensor onto the *returned* XComArg adds an
    # edge to ``deflate_sharpe`` only.  ``run_strategies`` was left with no
    # upstream at all, so the backtests started immediately, in parallel with the
    # sensor that was supposed to gate them -- against whatever predictions the
    # previous week had left behind.  The sensor must gate the work, not the
    # summary of the work.
    strategies = run_strategies()
    wait_for_training >> strategies
    deflate_sharpe(strategies)


backtesting()

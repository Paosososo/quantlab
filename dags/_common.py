"""Shared Airflow configuration.

Design rules for every DAG in this folder
------------------------------------------
1.  **A DAG file contains no business logic.**  Every task body is one call into
    :mod:`quantlab.pipelines` plus a return.  Logic that lives inside a DAG can
    only be run by Airflow, which means it cannot be unit tested, cannot be
    debugged locally, and cannot be reused by the API or a notebook.

2.  **Tasks are idempotent, so retries are safe.**  The pipeline functions use
    content-addressed raw storage and ``ON CONFLICT DO UPDATE`` loads, so
    re-running a task converges to the same database state.  That is what makes
    ``retries`` a real recovery mechanism rather than a way to write the same
    rows three times.

3.  **Incremental state comes from the database, not from the scheduler.**
    Loaders resume from the maximum timestamp already stored, minus an overlap
    window.  A backfill, a retry and a normal run therefore all converge, and a
    cleared task does not leave a hole.

4.  **`catchup=False`.**  These pipelines pull whole histories from their
    providers rather than one day per run, so replaying a year of daily
    execution dates would do the same work 250 times.  A backfill is a manual
    ``quantlab ingest-prices --start ...``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

#: Airflow retries transient provider failures.  Five minutes is long enough for
#: a rate limit to clear and short enough that a daily pipeline still finishes.
DEFAULT_ARGS: dict[str, Any] = {
    "owner": "quantlab",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": dt.timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": dt.timedelta(minutes=30),
    "execution_timeout": dt.timedelta(minutes=45),
    "email_on_failure": False,
    "email_on_retry": False,
}

#: US equity data is published after the 16:00 New York close.  Running at 23:00
#: UTC leaves a comfortable margin in both daylight-saving regimes without
#: waiting until the next calendar day.
DAILY_SCHEDULE = "0 23 * * 1-5"

#: A plain timezone-aware datetime rather than ``pendulum.datetime``.  Airflow
#: accepts either, and using the standard library keeps this module importable
#: without Airflow installed -- which is what lets the DAG conventions be unit
#: tested in ordinary CI.
START_DATE = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)

DAG_TAGS = ["quantlab"]


def dag_defaults(execution_timeout_minutes: int | None = None, **overrides: Any) -> dict[str, Any]:
    """Keyword arguments shared by every DAG.

    ``execution_timeout_minutes`` overrides the shared per-task timeout for a
    DAG whose work legitimately takes longer (model training, mainly).
    """
    default_args = dict(DEFAULT_ARGS)
    if execution_timeout_minutes is not None:
        default_args["execution_timeout"] = dt.timedelta(minutes=execution_timeout_minutes)
    base: dict[str, Any] = {
        "default_args": default_args,
        "start_date": START_DATE,
        "catchup": False,
        "max_active_runs": 1,
        "tags": DAG_TAGS,
    }
    base.update(overrides)
    return base


def summarise(payload: dict[str, Any], limit: int = 2_000) -> dict[str, Any]:
    """Trim a pipeline result before it goes into XCom.

    XCom values live in the metadata database; pushing a full metrics dump for
    every model into it turns the scheduler database into a data warehouse.
    Downstream tasks re-read what they need from PostgreSQL instead.
    """
    text = str(payload)
    if len(text) <= limit:
        return payload
    return {"truncated": True, "keys": sorted(payload), "preview": text[:limit]}


__all__ = ["DAG_TAGS", "DAILY_SCHEDULE", "DEFAULT_ARGS", "START_DATE", "dag_defaults", "summarise"]

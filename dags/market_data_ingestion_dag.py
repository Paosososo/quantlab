"""Daily market-data ingestion.

    ingest_prices ─┐
                   ├─> materialise_returns
    ingest_macro ──┘

Prices and macro run in parallel because they hit different providers and
neither depends on the other.  Return materialisation waits for prices; it does
not need macro, but joining the two here keeps the DAG a single readable graph
rather than two disconnected islands.
"""

from __future__ import annotations

from typing import Any

from _common import DAILY_SCHEDULE, dag_defaults, summarise
from airflow.decorators import dag, task


@dag(
    dag_id="market_data_ingestion",
    description="Fetch daily bars and macroeconomic series, then derive returns",
    schedule=DAILY_SCHEDULE,
    **dag_defaults(),
)
def market_data_ingestion() -> None:
    @task
    def ingest_prices() -> dict[str, Any]:
        from quantlab.pipelines import run_price_ingestion

        return summarise(run_price_ingestion())

    @task
    def ingest_macro() -> dict[str, Any]:
        from quantlab.pipelines import run_macro_ingestion

        return summarise(run_macro_ingestion())

    @task
    def materialise_returns(price_result: dict[str, Any]) -> dict[str, Any]:
        from quantlab.pipelines import run_return_materialisation

        # Only symbols that actually loaded; a failed ticker should not silently
        # produce an empty return series that later looks like real data.
        symbols = price_result.get("succeeded") if isinstance(price_result, dict) else None
        return summarise(run_return_materialisation(symbols))

    prices = ingest_prices()
    macro = ingest_macro()
    returns = materialise_returns(prices)
    macro >> returns  # ordering only: returns do not consume macro output


market_data_ingestion()

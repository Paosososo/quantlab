"""HTTP client the dashboard uses to reach the API.

Why the dashboard goes through the API rather than straight to PostgreSQL
------------------------------------------------------------------------
Reading the database directly would be less code and one fewer moving part.  The
API is used anyway for two reasons:

1.  It is the contract external consumers get.  A dashboard built on the same
    endpoints exercises them continuously, so a broken response model or a
    missing filter is found by looking at a chart rather than by a user report.
2.  It keeps the dashboard free of database credentials.  A read-only UI that
    holds a PostgreSQL password is a credential in one more place for no benefit.

The cost is that the dashboard is useless when the API is down, so the failure is
made explicit rather than silently rendering an empty page.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

DEFAULT_BASE_URL = os.environ.get("QUANTLAB_API_URL", "http://localhost:8000")


class ApiError(RuntimeError):
    """The API is unreachable or returned an error."""


class QuantlabClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    # -- plumbing ----------------------------------------------------------
    def _get(self, path: str, **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        try:
            response = self._client.get(path, params=clean)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the API at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise ApiError(f"{path} returned {response.status_code}: {response.text[:200]}")
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        try:
            response = self._client.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise ApiError(f"cannot reach the API at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise ApiError(f"{path} returned {response.status_code}: {response.text[:200]}")
        return response.json()

    @staticmethod
    def _page_to_frame(payload: dict[str, Any], time_columns: tuple[str, ...] = ()) -> pd.DataFrame:
        frame = pd.DataFrame(payload.get("items", []))
        for column in time_columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    # -- endpoints ---------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def assets(self) -> pd.DataFrame:
        return self._page_to_frame(self._get("/assets", limit=500))

    def prices(self, symbol: str, as_of: str | None = None) -> pd.DataFrame:
        payload = self._get(f"/assets/{symbol}/prices", limit=10_000, as_of=as_of)
        return self._page_to_frame(payload, ("ts", "available_at"))

    def backtests(self) -> pd.DataFrame:
        return self._page_to_frame(self._get("/backtests", limit=200))

    def equity_curve(self, backtest_id: int) -> pd.DataFrame:
        payload = self._get(f"/backtests/{backtest_id}/equity", limit=20_000)
        return self._page_to_frame(payload, ("ts",))

    def trades(self, backtest_id: int) -> pd.DataFrame:
        payload = self._get(f"/backtests/{backtest_id}/trades", limit=20_000)
        return self._page_to_frame(payload, ("decision_ts", "execution_ts"))

    def performance(self, backtest_id: int) -> dict[str, Any]:
        return self._get(f"/backtests/{backtest_id}/performance")

    def model_comparison(self) -> pd.DataFrame:
        return pd.DataFrame(self._get("/model-runs/comparison", limit=50))

    def model_runs(self) -> pd.DataFrame:
        return self._page_to_frame(self._get("/model-runs", limit=200))

    def feature_sets(self) -> pd.DataFrame:
        return pd.DataFrame(self._get("/feature-sets"))

    def optimise(self, symbols: list[str], method: str, lookback_days: int = 756) -> dict[str, Any]:
        return self._post(
            "/analytics/optimise",
            {"symbols": symbols, "method": method, "lookback_days": lookback_days},
        )


__all__ = ["DEFAULT_BASE_URL", "ApiError", "QuantlabClient"]

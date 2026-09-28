"""Dashboard helpers.

Only the pure functions are tested.  Streamlit's rendering is exercised by
running the app, not by unit tests, so the code that deserves tests was kept out
of ``app.py`` and put in ``formatting.py`` and ``client.py``.
"""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest

from quantlab.dashboard.client import ApiError, QuantlabClient
from quantlab.dashboard.formatting import (
    drawdown_frame,
    format_value,
    interpret_sharpe_inference,
    metrics_rows,
    monthly_returns_table,
    normalise_curves,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def equity() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    path = 100_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(index)))
    bench = 50_000 * np.cumprod(1 + rng.normal(0.0003, 0.009, len(index)))
    return pd.DataFrame({"ts": index, "equity": path, "benchmark_equity": bench})


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "kind", "expected"),
        [
            (0.1234, "pct", "12.34%"),
            (1.5, "num", "1.500"),
            # 1234.6 rather than 1234.5: Python rounds halves to even, so the
            # tie case is a property of the formatter, not of this function.
            (1234.6, "money", "1,235"),
            (42, "int", "42"),
            (None, "pct", "n/a"),
            (float("nan"), "num", "n/a"),
        ],
    )
    def test_values_format_predictably(self, value, kind, expected):
        assert format_value(value, kind) == expected

    def test_metrics_rows_skip_absent_keys(self):
        rows = metrics_rows({"sharpe_ratio": 1.2, "total_return": 0.4})
        labels = [label for label, _ in rows]
        assert labels == ["Total return", "Sharpe"]

    def test_metrics_rows_preserve_display_order(self):
        rows = metrics_rows({"max_drawdown": -0.2, "total_return": 0.5, "sharpe_ratio": 1.0})
        assert [label for label, _ in rows] == ["Total return", "Sharpe", "Max drawdown"]


class TestCurves:
    def test_both_curves_start_at_one_hundred(self, equity):
        frame = normalise_curves(equity)
        assert set(frame.columns) == {"Strategy", "Benchmark"}
        assert frame["Strategy"].iloc[0] == pytest.approx(100.0)
        assert frame["Benchmark"].iloc[0] == pytest.approx(100.0)

    def test_rebasing_preserves_relative_growth(self, equity):
        frame = normalise_curves(equity)
        raw_growth = equity["equity"].iloc[-1] / equity["equity"].iloc[0]
        assert frame["Strategy"].iloc[-1] / 100.0 == pytest.approx(raw_growth)

    def test_a_missing_benchmark_is_omitted(self, equity):
        frame = normalise_curves(equity.drop(columns=["benchmark_equity"]))
        assert list(frame.columns) == ["Strategy"]

    def test_empty_input_returns_empty(self):
        assert normalise_curves(pd.DataFrame()).empty

    def test_drawdown_is_never_positive(self, equity):
        drawdown = drawdown_frame(equity)
        assert (drawdown <= 1e-12).all()
        assert drawdown.iloc[0] == pytest.approx(0.0)

    def test_monthly_table_has_month_columns(self, equity):
        table = monthly_returns_table(equity)
        assert not table.empty
        assert set(table.columns) <= {
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        }
        assert table.index.name == "year"


class TestSharpeInterpretation:
    def test_missing_inference_is_reported_plainly(self):
        assert "Not enough observations" in interpret_sharpe_inference(None)

    def test_deflation_is_mentioned_when_present(self):
        text = interpret_sharpe_inference(
            {
                "probabilistic_sharpe": 0.9,
                "deflated_sharpe": 0.3,
                "n_trials": 50,
                "minimum_track_record_years": 4.2,
            }
        )
        assert "90%" in text
        assert "50 strategy configurations" in text
        assert "30%" in text
        assert "4.2 years" in text

    def test_an_unattainable_track_record_is_stated_honestly(self):
        text = interpret_sharpe_inference(
            {"probabilistic_sharpe": 0.4, "minimum_track_record_years": None}
        )
        assert "No track record length" in text


class TestClient:
    @staticmethod
    def _client(handler) -> QuantlabClient:
        client = QuantlabClient("http://test")
        client._client = httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(handler)
        )
        return client

    def test_pages_become_dataframes_with_parsed_timestamps(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "symbol": "A",
                            "ts": "2020-01-01T00:00:00Z",
                            "available_at": "2020-01-01T21:00:00Z",
                            "close": 100.0,
                        }
                    ],
                    "limit": 10,
                    "offset": 0,
                    "returned": 1,
                    "total": 1,
                },
            )

        frame = self._client(handler).prices("A")
        assert len(frame) == 1
        assert str(frame["ts"].dt.tz) == "UTC"

    def test_an_http_error_becomes_an_api_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        with pytest.raises(ApiError, match="returned 500"):
            self._client(handler).assets()

    def test_an_unreachable_api_becomes_an_api_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(ApiError, match="cannot reach the API"):
            self._client(handler).health()

    def test_none_parameters_are_dropped_from_the_query(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"items": [], "limit": 1, "offset": 0, "returned": 0})

        self._client(handler).prices("A", as_of=None)
        assert "as_of" not in seen[0]

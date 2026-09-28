"""Streamlit dashboard.

Streamlit rather than Dash or a React front end.  The requirement was a simple
professional dashboard without spending effort on visual design, and Streamlit
turns a page of Python into a working UI.  A React front end would be a second
codebase, a build step and a deployment target, for charts that nobody is going
to style.

Everything is read through the REST API, so the dashboard is a client of the
same contract an external consumer would use; see ``client.py``.

Run with:  streamlit run src/quantlab/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab.dashboard.client import DEFAULT_BASE_URL, ApiError, QuantlabClient
from quantlab.dashboard.formatting import (
    drawdown_frame,
    format_value,
    interpret_sharpe_inference,
    metrics_rows,
    monthly_returns_table,
    normalise_curves,
)

st.set_page_config(page_title="quantlab", page_icon="::", layout="wide")

STRATEGY_COLOUR = "#2563eb"
BENCHMARK_COLOUR = "#94a3b8"
NEGATIVE_COLOUR = "#dc2626"


@st.cache_resource
def get_client(base_url: str) -> QuantlabClient:
    return QuantlabClient(base_url)


@st.cache_data(ttl=120)
def load_assets(base_url: str) -> pd.DataFrame:
    return get_client(base_url).assets()


@st.cache_data(ttl=120)
def load_prices(base_url: str, symbol: str) -> pd.DataFrame:
    return get_client(base_url).prices(symbol)


@st.cache_data(ttl=120)
def load_backtests(base_url: str) -> pd.DataFrame:
    return get_client(base_url).backtests()


@st.cache_data(ttl=120)
def load_equity(base_url: str, backtest_id: int) -> pd.DataFrame:
    return get_client(base_url).equity_curve(backtest_id)


@st.cache_data(ttl=120)
def load_trades(base_url: str, backtest_id: int) -> pd.DataFrame:
    return get_client(base_url).trades(backtest_id)


@st.cache_data(ttl=120)
def load_performance(base_url: str, backtest_id: int) -> dict:
    return get_client(base_url).performance(backtest_id)


@st.cache_data(ttl=120)
def load_model_comparison(base_url: str) -> pd.DataFrame:
    return get_client(base_url).model_comparison()


def line_chart(frame: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    colours = {"Strategy": STRATEGY_COLOUR, "Benchmark": BENCHMARK_COLOUR}
    for column in frame.columns:
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                name=column,
                mode="lines",
                line={"width": 2, "color": colours.get(column)},
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=y_title,
        hovermode="x unified",
        height=420,
        margin={"l": 40, "r": 20, "t": 50, "b": 30},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    return figure


def render_overview(client_url: str) -> None:
    st.header("Market data")
    assets = load_assets(client_url)
    if assets.empty:
        st.info("No assets loaded yet. Run `quantlab ingest-prices` first.")
        return

    symbol = st.selectbox("Asset", assets["symbol"].tolist())
    prices = load_prices(client_url, symbol)
    if prices.empty:
        st.warning(f"No price history stored for {symbol}.")
        return

    frame = prices.set_index("ts").sort_index()
    left, right = st.columns([3, 1])
    with left:
        st.plotly_chart(
            line_chart(
                frame[["adj_close"]].rename(columns={"adj_close": "Strategy"}),
                f"{symbol} adjusted close",
                "price",
            ),
            width="stretch",
        )
    with right:
        returns = frame["adj_close"].pct_change(fill_method=None).dropna()
        st.metric("Observations", f"{len(frame):,}")
        st.metric("First bar", str(frame.index.min().date()))
        st.metric("Last bar", str(frame.index.max().date()))
        if len(returns) > 2:
            st.metric("Annualised volatility", format_value(returns.std() * (252**0.5), "pct"))

    with st.expander("Point-in-time availability"):
        st.caption(
            "Every row carries `available_at`, the instant the observation became "
            "public. Features and backtests filter on it, which is what makes "
            "look-ahead bias structurally hard rather than a thing to remember."
        )
        st.dataframe(
            prices[["ts", "available_at", "close", "adj_close", "volume"]].tail(10),
            width="stretch",
        )


def render_backtests(client_url: str) -> None:
    st.header("Backtests")
    backtests = load_backtests(client_url)
    if backtests.empty:
        st.info("No backtests stored yet. Run `quantlab backtest`.")
        return

    labels = {
        int(
            row["id"]
        ): f"#{row['id']} {row['strategy_name']} ({row['start_ts'][:10]} to {row['end_ts'][:10]})"
        for _, row in backtests.iterrows()
    }
    backtest_id = st.selectbox("Backtest", list(labels), format_func=lambda i: labels[i])
    equity = load_equity(client_url, backtest_id)
    if equity.empty:
        st.warning("This backtest has no stored equity curve.")
        return

    performance = load_performance(client_url, backtest_id)
    metrics = performance.get("metrics", {})

    rows = metrics_rows(metrics)
    for chunk_start in range(0, len(rows), 5):
        for column, (label, value) in zip(
            st.columns(5), rows[chunk_start : chunk_start + 5], strict=False
        ):
            column.metric(label, value)

    st.plotly_chart(
        line_chart(normalise_curves(equity), "Equity curve, rebased to 100", "index"),
        width="stretch",
    )

    drawdown = drawdown_frame(equity)
    figure = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.to_numpy(),
            fill="tozeroy",
            mode="lines",
            line={"color": NEGATIVE_COLOUR, "width": 1},
            name="Drawdown",
        )
    )
    figure.update_layout(
        title="Drawdown",
        yaxis_tickformat=".0%",
        height=280,
        margin={"l": 40, "r": 20, "t": 50, "b": 30},
    )
    st.plotly_chart(figure, width="stretch")

    st.subheader("How much to trust this")
    st.write(interpret_sharpe_inference(performance.get("sharpe_inference")))

    tab_monthly, tab_risk, tab_trades, tab_config = st.tabs(
        ["Monthly returns", "Risk", "Trades", "Configuration"]
    )
    with tab_monthly:
        table = monthly_returns_table(equity)
        if table.empty:
            st.info("Not enough history for a monthly breakdown.")
        else:
            st.dataframe(table.style.format("{:.2%}", na_rep=""), width="stretch")
    with tab_risk:
        risk = performance.get("risk", {})
        if risk:
            st.dataframe(
                pd.DataFrame({"metric": list(risk), "value": list(risk.values())}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Not enough observations for a risk summary.")
    with tab_trades:
        trades = load_trades(client_url, backtest_id)
        if trades.empty:
            st.info("This strategy placed no trades.")
        else:
            st.caption(
                f"{len(trades):,} fills. `execution_ts` is always strictly after "
                "`decision_ts`; the database enforces it with a CHECK constraint."
            )
            st.dataframe(trades.tail(200), width="stretch", hide_index=True)
    with tab_config:
        selected = backtests.loc[backtests["id"] == backtest_id].iloc[0]
        st.json(
            {
                "strategy": selected["strategy_name"],
                "kind": selected["strategy_kind"],
                "execution_timing": selected["execution_timing"],
                "initial_cash": selected["initial_cash"],
                "benchmark": selected.get("benchmark_symbol"),
                "config_hash": selected["config_hash"],
            }
        )


def render_models(client_url: str) -> None:
    st.header("Model results")
    st.caption(
        "The research question: do machine-learning models beat simple statistical "
        "baselines after realistic costs? `r2_oos_vs_zero` is the number that "
        "answers it. Values at or below zero mean the model is not beating a "
        "forecast of no change."
    )
    comparison = load_model_comparison(client_url)
    if comparison.empty:
        st.info("No model runs stored yet. Run `quantlab train`.")
        return

    st.dataframe(
        comparison.style.format(
            {
                "rmse": "{:.5f}",
                "r2_oos_vs_zero": "{:.5f}",
                "information_coefficient": "{:.4f}",
                "directional_accuracy": "{:.2%}",
            },
            na_rep="n/a",
        ),
        width="stretch",
        hide_index=True,
    )

    plottable = comparison.dropna(subset=["r2_oos_vs_zero"])
    if not plottable.empty:
        figure = go.Figure(
            go.Bar(
                x=plottable["model_type"],
                y=plottable["r2_oos_vs_zero"],
                marker_color=[
                    STRATEGY_COLOUR if v > 0 else NEGATIVE_COLOUR
                    for v in plottable["r2_oos_vs_zero"]
                ],
            )
        )
        figure.add_hline(y=0.0, line_dash="dash", line_color="#475569")
        figure.update_layout(
            title="Out-of-sample R-squared against a zero forecast",
            height=360,
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        )
        st.plotly_chart(figure, width="stretch")


def main() -> None:
    st.sidebar.title("quantlab")
    base_url = st.sidebar.text_input("API base URL", value=DEFAULT_BASE_URL)

    try:
        health = get_client(base_url).health()
    except ApiError as exc:
        st.error(f"Cannot reach the API.\n\n{exc}")
        st.info("Start it with `uvicorn quantlab.api.main:app --reload`, or `docker compose up`.")
        st.stop()
        return

    st.sidebar.success(f"API {health['version']} ({health['environment']})")
    if health.get("database") != "ok":
        st.sidebar.error("Database unavailable")

    page = st.sidebar.radio("View", ["Market data", "Backtests", "Models"])
    st.sidebar.caption(
        "Read-only. Ingestion, training and backtesting run in Airflow or through "
        "the `quantlab` CLI."
    )

    if page == "Market data":
        render_overview(base_url)
    elif page == "Backtests":
        render_backtests(base_url)
    else:
        render_models(base_url)


if __name__ == "__main__":
    main()
else:  # Streamlit executes the module rather than importing it as __main__
    main()

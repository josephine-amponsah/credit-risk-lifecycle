"""Model backtesting: expected vs realised loss and default rate by vintage."""
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from services.api_client import get_backtest

dash.register_page(__name__, path="/backtesting", name="Model Backtesting")

layout = html.Div([
    html.P(
        "Predicted vs observed default rates by issue quarter. Vintages up to 2014Q4 were used to "
        "calibrate PDs; later vintages are out-of-time. Recent vintages are still running, so their "
        "observed defaults are incomplete (see share active).",
        className="text-muted",
    ),
    dbc.Card(dbc.CardBody(dcc.Graph(id="backtest-chart", config={"displayModeBar": False},
                                    style={"height": "380px"})), className="shadow-sm border-0 mb-3"),
    dbc.Card(dbc.CardBody(html.Div(id="backtest-table")), className="shadow-sm border-0"),
    dcc.Interval(id="backtest-load", interval=1, max_intervals=1),
])


@callback(Output("backtest-chart", "figure"), Output("backtest-table", "children"),
          Input("backtest-load", "n_intervals"))
def render(_):
    try:
        df = pd.DataFrame(get_backtest()["data"])
    except Exception as exc:  # show the problem instead of an empty page
        return go.Figure(), dbc.Alert(f"Backtest unavailable: {exc}", color="warning")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["vintage"], y=df["predicted_default_rate"], name="Predicted (mean PD)", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=df["vintage"], y=df["observed_default_rate"], name="Observed", mode="lines+markers"))
    fig.add_trace(go.Bar(x=df["vintage"], y=df["share_active"], name="Share still active", opacity=0.25, yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Default rate", tickformat=".0%"),
        yaxis2=dict(title="Share active", tickformat=".0%", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", y=1.12), margin=dict(l=40, r=40, t=40, b=40),
    )

    table = df[["vintage", "n_loans", "predicted_default_rate", "observed_default_rate",
                "expected_loss", "realised_loss", "loss_ratio", "share_active"]].copy()
    for col in ["predicted_default_rate", "observed_default_rate", "share_active"]:
        table[col] = (table[col] * 100).round(1).astype(str) + "%"
    for col in ["expected_loss", "realised_loss"]:
        table[col] = (table[col] / 1e6).round(1).astype(str) + "m"
    table["loss_ratio"] = table["loss_ratio"].round(2)
    return fig, dash_table.DataTable(table.to_dict("records"), page_size=12,
                                     style_cell={"fontFamily": "inherit", "padding": "6px"})

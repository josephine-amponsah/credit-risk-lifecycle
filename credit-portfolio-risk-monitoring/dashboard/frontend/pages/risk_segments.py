"""Credit risk: expected loss and one-factor (Vasicek) credit VaR for a vintage."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from services.api_client import get_expected_loss, get_var

dash.register_page(__name__, path="/risk_segments", name="Credit Risk & VaR")


def card(title, value_id):
    return dbc.Col(dbc.Card(dbc.CardBody([html.H6(title, className="card-title"),
                                          html.H4(id=value_id, className="mb-0")]),
                            className="h-100 shadow-sm border-0"), width=3)


layout = html.Div([
    dcc.Store(id="sales-data"),  # triggers the shared period-dropdown loader in pages/overview.py
    dbc.Row([
        dbc.Col([html.Label("Period", className="form-label"),
                 dcc.Dropdown(id="date-time-filter", placeholder="Select period", clearable=True)], width=3),
        dbc.Col([html.Label("Asset correlation (rho)", className="form-label"),
                 dcc.Slider(id="rho-slider", min=0.01, max=0.20, step=0.01, value=0.05,
                            marks={0.03: "0.03", 0.05: "0.05", 0.10: "0.10", 0.15: "0.15", 0.20: "0.20"})], width=5),
    ], align="end", className="g-3 mb-3"),
    dbc.Row([
        card("Expected loss (PD × LGD × EAD)", "card-expected-loss"),
        card("Credit VaR, 99%", "card-var"),
        card("Unexpected loss (VaR − EL)", "card-ul"),
    ], className="g-3 mb-3"),
    dbc.Card(dbc.CardBody(dcc.Graph(id="risk-dist-chart", config={"displayModeBar": False},
                                    style={"height": "360px"})), className="shadow-sm border-0"),
])


@callback([Output("card-expected-loss", "children"), Output("card-var", "children"),
           Output("card-ul", "children"), Output("risk-dist-chart", "figure")],
          [Input("date-time-filter", "value"), Input("rho-slider", "value")])
def update_risk(period, rho):
    try:
        el = get_expected_loss(period)["total_expected_loss"]
        var_payload = get_var(period, confidence=0.99, rho=rho)
    except Exception as exc:
        return "-", "-", "-", go.Figure(layout={"title": f"Risk data unavailable: {exc}"})

    var_val = var_payload["var"]
    bins, counts = var_payload["hist"]["bins"], var_payload["hist"]["counts"]
    fig = go.Figure()
    if bins and counts:
        centers = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))
        fig.add_trace(go.Bar(x=centers, y=counts, name="Simulated portfolio loss"))
        fig.add_vline(x=el, line_dash="dot", annotation_text="Expected loss", annotation_position="top left")
        fig.add_vline(x=var_val, line_dash="dash", line_color="red",
                      annotation_text="99% VaR", annotation_position="top right")
        fig.update_layout(xaxis_title="Portfolio loss", yaxis_title="Scenarios", showlegend=False)
    return f"{el:,.0f}", f"{var_val:,.0f}", f"{var_val - el:,.0f}", fig

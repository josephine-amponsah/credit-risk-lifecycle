import dash
from dash import Dash, html, dcc, Input, Output, callback, State
import plotly.express as px
from dash import dash_table
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
import sys
import json
import numerize
from numerize import numerize
import os
import io
import requests
import functools
from pathlib import Path

from flask_caching import Cache
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.css"

pio.templates["flatly"] = go.layout.Template(
    layout=dict(
        font=dict(color="#2C3E50", family="Open Sans, sans-serif"),
        paper_bgcolor="#F8F9FA",
        plot_bgcolor="#F8F9FA",
        title_font=dict(color="#2C3E50"),
        legend=dict(title_font=dict(color="#2C3E50"), font=dict(color="#2C3E50")),
        colorway=["#18BC9C", "#2C3E50", "#3498DB", "#F39C12", "#E74C3C", "#95A5A6"],
        xaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#CBD5E1", tickfont=dict(color="#2C3E50"), title_font=dict(color="#2C3E50")),
        yaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#CBD5E1", tickfont=dict(color="#2C3E50"), title_font=dict(color="#2C3E50")),
        margin=dict(l=40, r=20, t=40, b=40),
    )
)
pio.templates.default = "flatly"

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[
                dbc.themes.FLATLY, dbc.icons.BOOTSTRAP, dbc_css, dbc.icons.BOOTSTRAP, dbc.icons.FONT_AWESOME])

server = app.server


BACKEND_DATA_DIR = Path(__file__).resolve().parents[1] / 'backend' / 'app' / 'data'
timeout = 20

# Flask-Caching 2.x uses backend class names such as `SimpleCache`, not the older
# lowercase alias `simple` that resolves to `flask_caching.backends.simple`.
# Keep the default in-memory cache but normalize common env values for compatibility.
try:
    cache_type = os.environ.get('CACHE_TYPE', 'SimpleCache').strip()
    cache_alias_map = {
        'simple': 'SimpleCache',
        'simplecache': 'SimpleCache',
        'null': 'NullCache',
        'none': 'NullCache',
    }
    cache_type = cache_alias_map.get(cache_type.lower(), cache_type)
    cache = Cache(server, config={'CACHE_TYPE': cache_type})
except Exception as e:
    # avoid failing app startup due to missing cache backends
    print("[warning] flask-caching init failed, falling back to lru cache:", e, file=sys.stderr)
    cache = None

# Ensure `cache.memoize` exists so code using it as a decorator doesn't crash.
if cache is None:
    class _CacheStub:
        def memoize(self, timeout=None):
            def _decorator(func):
                return functools.lru_cache(maxsize=32)(func)
            return _decorator
    cache = _CacheStub()


# def _app_data_impl(period: str | None = None):
#     try:
#         if not BACKEND_DATA_DIR.exists():
#             return json.dumps({})

#         files = sorted(BACKEND_DATA_DIR.glob('data_*.parquet'))
#         if not files:
#             files = sorted(BACKEND_DATA_DIR.glob('data_*.csv'))
#         if not files:
#             return json.dumps({})

#         if period:
#             matches = [p for p in files if p.stem == f'data_{period}']
#             if matches:
#                 target = matches[0]
#             else:
#                 return json.dumps({})
#         else:
#             target = files[-1]

#         try:
#             df = pd.read_parquet(target)
#         except Exception:
#             df = pd.read_csv(target, parse_dates=['issue_d'], low_memory=False)

#         for c in df.select_dtypes(include=['object']).columns:
#             df[c] = df[c].apply(lambda v: None if pd.isna(v) else (v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v)))
#         return df.to_json(date_format='iso')
#     except Exception:
#         return json.dumps({})


# # Apply caching: prefer Flask-Caching if initialized, otherwise use functools.lru_cache
# if cache:
#     app_data = cache.memoize(timeout=timeout)(_app_data_impl)
# else:
#     app_data = functools.lru_cache(maxsize=32)(_app_data_impl)
    
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    # dcc.Store(id ="sales-store", data = app_data()),
    dbc.Row([
            html.Nav([
                html.Div([
                    html.A("Portfolio Monitor", className = "navbar-brand"),
                    html.Div([
                        html.Ul([
                        html.Li([
                            html.A("Insights", id="overview-link", className="nav-link", href=dash.page_registry['pages.overview']['path'])
                            ], id="overview-item", className="nav-item"),
                        html.Li(
                                [
                            html.A("Credit Risk & VaR", id="risk-segments-link", className="nav-link", href=dash.page_registry['pages.risk_segments']['path'])
                                ], id="risk-segments-item", className = "nav-item"),
                        html.Li(
                                [
                            html.A("Survival Analysis", id="survival-analysis-link", className="nav-link", href=dash.page_registry['pages.survival_analysis']['path'])
                                ], id="survival-analysis-item", className = "nav-item"),
                        html.Li(
                                [
                             html.A("Model Backtesting", id="backtesting-link", className="nav-link", href=dash.page_registry['pages.npv_backtesting']['path'])
                                ], id="backtesting-item", className = "nav-item") 
                        #,
                        # html.Li(
                        #         [
                        #     html.A("Reports", className ="nav-link", href= dash.page_registry['pages.documentation']['path'])
                        #         ], className = "nav-item"),
                    ], className="navbar-nav me-auto"
                    )
                    ], className = "collapse navbar-collapse" ,id="navbarColor01"),
                ], className= "container-fluid"),
            ],
            className = "navbar navbar-expand-lg bg-primary",
            **{"data-bs-theme": "dark"}),
            html.Br(),
            html.Div(dbc.Container(dash.page_container, fluid=True), className="page-shell px-3")
    ], className = "")
]
)


@app.callback(
    Output("overview-item", "className"),
    Output("risk-segments-item", "className"),
    Output("survival-analysis-item", "className"),
    Output("backtesting-item", "className"),
    Output("overview-link", "className"),
    Output("risk-segments-link", "className"),
    Output("survival-analysis-link", "className"),
    Output("backtesting-link", "className"),
    Input("url", "pathname"),
)
def update_nav_classes(pathname):
    current_path = pathname or "/"

    def item_class(target):
        return "nav-item active" if current_path == target else "nav-item"

    def link_class(target):
        return "nav-link active" if current_path == target else "nav-link"

    return (
        item_class(dash.page_registry['pages.overview']['path']),
        item_class(dash.page_registry['pages.risk_segments']['path']),
        item_class(dash.page_registry['pages.survival_analysis']['path']),
        item_class(dash.page_registry['pages.npv_backtesting']['path']),
        link_class(dash.page_registry['pages.overview']['path']),
        link_class(dash.page_registry['pages.risk_segments']['path']),
        link_class(dash.page_registry['pages.survival_analysis']['path']),
        link_class(dash.page_registry['pages.npv_backtesting']['path']),
    )


if __name__ == '__main__':
    # model = joblib.load("modules/forecaster.pkl")
    app.run(debug=os.getenv('DASH_DEBUG') == '1', host='0.0.0.0', port=int(os.getenv('PORT', 8050)))
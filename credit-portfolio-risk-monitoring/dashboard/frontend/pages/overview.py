import dash
from dash import html, dcc, Input, Output, callback
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from numerize import numerize

from services.api_client import get_summary, get_aggregates, get_options, get_periods

# app = Dash(__name__)
dash.register_page(__name__, path = "/")

layout = html.Div([
    dcc.Store(id="sales-data"),
    dbc.Row([
        dbc.Col([
            html.Label('Period', className='form-label'),
            dcc.Dropdown(
                placeholder='Select period',
                id='date-time-filter',
                clearable=True,
                searchable=True,
                style={'borderRadius': '0.5rem'}
            )
        ], width=3),
        dbc.Col([
            html.Label('Loan purpose', className='form-label'),
            dcc.Dropdown(
                placeholder='Select loan purpose',
                id='loan-purpose-filter',
                multi=True,
                clearable=True,
                searchable=True,
                style={'borderRadius': '0.5rem'}
            )
        ], width=4),
        dbc.Col([
            html.Label('Actions', className='form-label'),
            dbc.Button('Download Risk Report', id='downloader', color='primary', className='w-100')
        ], width=2),
    ], align='end', className='g-3 mb-3'),

    html.Br(),

    # summary cards
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Total Lent", className="card-title"), html.H4(id='card-total-lent', className='mb-0')])], className='h-100 shadow-sm border-0'), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Repayments", className="card-title"), html.H4(id='card-repaid', className='mb-0')])], className='h-100 shadow-sm border-0'), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Default Rate", className="card-title"), html.H4(id='card-default-rate', className='mb-0')])], className='h-100 shadow-sm border-0'), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H6("Interest Earned", className="card-title"), html.H4(id='card-interest', className='mb-0')])], className='h-100 shadow-sm border-0'), width=3),
    ], className='g-3 mb-3'),

    # bar chart
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody(dcc.Graph(id='overview-bar-chart', config={'displayModeBar': False}, style={'height': '360px'}))], className='shadow-sm border-0'), width=12)
    ]),

    html.Br()
])

# Summary cards: call API for totals
@callback(
    [Output('card-total-lent', 'children'), Output('card-repaid', 'children'), Output('card-default-rate', 'children'), Output('card-interest', 'children')],
    [Input('date-time-filter', 'value'), Input('loan-purpose-filter', 'value')]
)
def update_summary(period, loan_purposes):
    try:
        payload = get_summary(period, loan_purposes)
        total = numerize.numerize(payload.get('total_lent', 0))
        repaid = numerize.numerize(payload.get('repaid', 0))
        default_rate = f"{payload.get('default_rate', 0):.2%}"
        interest = numerize.numerize(payload.get('interest_earned', 0))
        return total, repaid, default_rate, interest
    except Exception:
        return "-", "-", "-", "-"


# Bar chart aggregates
@callback(
    Output('overview-bar-chart', 'figure'),
    [Input('date-time-filter', 'value'), Input('loan-purpose-filter', 'value')]
)
def update_bar(period, loan_purposes):
    try:
        payload = get_aggregates(period, loan_purposes)
        data = payload.get('data', []) if isinstance(payload, dict) else payload
        if not data:
            return go.Figure()
        df = pd.DataFrame(data)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Lent', x=df['loan_purpose'], y=df['lent']))
        fig.add_trace(go.Bar(name='Repaid', x=df['loan_purpose'], y=df['repaid']))
        fig.add_trace(go.Bar(name='Interest', x=df['loan_purpose'], y=df['interest']))
        fig.update_layout(barmode='group', xaxis_title='Loan Purpose', yaxis_title='Amount')
        return fig
    except Exception:
        return go.Figure()


# Populate loan purpose options from API; default select all
@callback(
    Output('loan-purpose-filter', 'options'),
    Output('loan-purpose-filter', 'value'),
    Input('date-time-filter', 'value')
)
def load_loan_purposes(period):
    try:
        opts = get_options(period).get('loan_purposes', [])
        options = [{'label': o, 'value': o} for o in opts]
        values = [o['value'] for o in options]
        return options, values
    except Exception:
        return [], []


# Populate period dropdown on page load (uses the API to list available quarters)
@callback(
    Output('date-time-filter', 'options'),
    Output('date-time-filter', 'value'),
    Input('sales-data', 'data')
)
def load_periods(_store_data):
    try:
        periods = get_periods().get('periods', [])
        options = [{'label': p, 'value': p} for p in periods]
        value = periods[-1] if periods else None
        return options, value
    except Exception:
        return [], None



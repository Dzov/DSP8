from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import functions as fn
import shap
import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.MINTY])

############################################################################################
##################                       COMPONENTS                     ####################
############################################################################################

navbar = dbc.Navbar(
    dbc.Container(
        [dbc.Col(html.Img(src=app.get_relative_path('/assets/logo.png'), height="70px"), md=10),
         dbc.Col(dcc.Dropdown(fn.get_clients(), placeholder='Select a client',
                              id='client-dropdown', style={'width': '100%'})),

         ]
    ),
    color="primary",
    dark=True,
)

placeholder = dbc.Container(
    dbc.Row(
        [
            dbc.Col(html.H1("Client Profile"), width=6,
                    className="offset-3 text-center"),
            dbc.Col(html.P(
                "Select a client ID from the dropdown menu to display information",
                className="lead",
            ), width=6, className="offset-3 text-center"),
        ]
    ),
    style={'margin-top': '20px'}
)


def get_client_feature_importance(client_id):
    expected_values, features, shap_values = fn.get_client_feature_importance(
        client_id)

    force_plot = shap.force_plot(expected_values[1], shap_values[1], features)
    force_plot_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    return html.Iframe(srcDoc=force_plot_html, style={"width": "100%"})


def get_global_importance():
    df = fn.get_global_feature_importance().transpose().reset_index()
    df.columns = ['Feature', 'Importance']
    df_plot = df.sort_values(by='Importance', ascending=True)

    fig = px.bar(df_plot, x='Importance', y='Feature', orientation='h',
                 labels={'Importance': 'Feature Importance'})

    return html.Iframe(srcDoc=fig.to_html(), style={"width": "100%", "height": "500px"})


def bar_comparison(client_id):
    client_data, neighbours_data = fn.get_client_neighbours(client_id)

    neighbours_data = pd.DataFrame(neighbours_data.mean()).transpose()
    df = pd.concat([client_data, neighbours_data], keys=[
        "Client", "Neighbours"])
    fig = go.Figure()
    for feature in df.columns:
        fig.add_trace(go.Bar(
            x=df.index.get_level_values(0),
            y=df[feature],
            name=feature,
        ))

    fig.update_traces(marker=dict(
        color=['#FFA88F' if index[0] == 'Client' else '#A1D8E4' for index in df.index]))

    fig.update_layout(barmode='group',
                      yaxis=dict(title='Feature Value'),
                      xaxis=dict(title='Comparison'),
                      showlegend=True)

    return html.Iframe(srcDoc=fig.to_html(), style={"width": "100%", "height": "500px"})


def build_comparison_graphs(client_id, selected_features):
    client_data, neighbours_data = fn.get_client_neighbours(client_id)
    fig = make_subplots(rows=len(selected_features), cols=1,
                        subplot_titles=selected_features)

    for i, feature in enumerate(selected_features, start=1):
        client_value = client_data[feature].values[0]
        neighbours_values = neighbours_data[feature]

        fig.add_trace(go.Histogram(x=neighbours_values, histnorm='probability density',
                      name=f'{feature} - Neighbours'), row=i, col=1)

        fig.add_shape(type="line", x0=client_value, y0=0, x1=client_value,
                      y1=1, line=dict(color="#FFA88F",), row=i, col=1)

    fig.update_layout(height=400 * len(selected_features), showlegend=False)
    return html.Iframe(srcDoc=fig.to_html(), style={"width": "100%", "height": "500px"})


def build_client_panel(client_info, prediction):
    general = dbc.Card([
        dbc.CardHeader(f"General Information: "),
        dbc.CardBody(
            [
                html.P(
                    [html.Strong("Age:"), f" {client_info['age']}"],
                    className="card-text",
                ),
                html.P(
                    [html.Strong("Gender:"), f" {client_info['gender']}"],
                    className="card-text",
                ),
                html.P(
                    [html.Strong("Family Status:"),
                     f" {client_info['maritalStatus']}"],
                    className="card-text",
                ),
                html.P(
                    [html.Strong("Children:"),
                     f" {client_info['childrenCount']}"],
                    className="card-text",
                ),

            ]
        )], color="primary",  outline=True)

    financial = dbc.Card([
        dbc.CardHeader(f"Financial Information: "),
        dbc.CardBody(
            [
                html.P(
                    [html.Strong("Occupation:"),
                     f" {client_info['occupation']}"],
                    className="card-text",
                ),
                html.P(
                    [html.Strong("Years Employed:"),
                     f" {client_info['workSeniority']}"],
                    className="card-text",
                ),
                html.P(
                    [html.Strong("Total Income:"),
                     f" {client_info['totalIncome']}"],
                    className="card-text",
                ),
                html.P(
                    [html.Strong("Credit:"), f" {client_info['credit']}"],
                    className="card-text",
                ),

            ]
        )], color="secondary",  outline=True)

    loan = dbc.Card([
        dbc.CardHeader(f"Loan Information: "),
        dbc.CardBody(
            [
                    html.P(
                        [html.Strong("Score: "), f" {prediction['probability'][0]}  ", dbc.Badge(
                            "Granted", color="success", className="me-1") if prediction['prediction'][0] == 0 else dbc.Badge("Denied", color="danger", className="me-1")],
                        className="card-text",
                    ),

                    ]
        )], color="info",  outline=True)

    return dbc.Container([
        dbc.Row(
            [
                dbc.Col(html.H1("Client Profile"), width=6,
                        className="offset-3 text-center"),
            ], style={'margin-bottom': '20px'}
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Card(general, color="primary", outline=True)),
                dbc.Col(dbc.Card(financial, color="secondary", outline=True)),
                dbc.Col(dbc.Card(loan, color="info", outline=True),  width=2),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(html.H2("Local Feature Importance"), width=6,
                        className="offset-3 text-center"),
                dbc.Col(get_client_feature_importance(
                    client_info['clientId']), width=12),

            ],
            className="mb-4", style={'margin-top': '40px'}
        ),
        dbc.Row(
            [
                dbc.Col(html.H2("Global Feature Importance"), width=6,
                        className="offset-3 text-center"),
                dbc.Col(get_global_importance(), width=12),
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(html.H3("Distribution comparison with 20 nearest neighbours"), width=6,
                        className="offset-3 text-center"),
                dbc.Col(
                    dcc.Dropdown(
                        id='feature-dropdown',
                        options=[{'label': feature, 'value': feature}
                                 for feature in fn.get_all_features()],
                        multi=True,
                        value=['CREDIT_INCOME_PERCENT'],
                        placeholder="Select features to compare"
                    )
                ),
                dbc.Col(html.Div(id='comparison-graphs'), width=12),
                dbc.Col(html.H3("Comparison to mean values of 20 nearest neighbours"), width=6,
                        className="offset-3 text-center"),
                dbc.Col(html.Div(bar_comparison(client_info['clientId'])), width=12),
            ],
            className="mb-4",
        ),
    ],
        style={'margin-top': '20px'}
    )


############################################################################################
##################                         LAYOUT                       ####################
############################################################################################


app.layout = dbc.Container(
    [
        navbar,
        html.Div(id='client_panel')
    ],
    fluid=True
)


@app.callback(
    Output('client_panel', 'children'),
    Input('client-dropdown', 'value'),
)
def get_client_info(client_id):
    if client_id is None:
        return placeholder
    client_info = fn.get_client_info(client_id)
    client_prediction = fn.get_prediction(client_id)

    return build_client_panel(client_info, client_prediction)


@app.callback(
    Output('comparison-graphs', 'children'),
    [Input('client-dropdown', 'value'),
     Input('feature-dropdown', 'value')]
)
def update_comparison_graphs(client_id, selected_features):
    if client_id is None or not selected_features:
        return html.Div()
    return build_comparison_graphs(client_id, selected_features)


if __name__ == '__main__':
    app.run_server(debug=True)

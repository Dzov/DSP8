from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import functions as fn
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.MINTY])
server = app.server

############################################################################################
##################                       COMPONENTS                     ####################
############################################################################################
app.title = 'Dashboard Prêt à Dépenser'

navbar = dbc.Navbar(
    dbc.Container(
        [dbc.Col(html.Img(src=app.get_relative_path('/assets/logo.png'), height="70px", alt='Logo Prêt à Dépenser'), md=10),
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
            dbc.Col(html.P(
                "Select a client ID from the dropdown menu to display information",
                className="lead",
            ), width=6, className="offset-3 text-center"),
        ]
    ),
)


def get_global_importance():
    df = fn.get_global_feature_importance().transpose().reset_index()
    df.columns = ['Feature', 'Importance']
    df_plot = df.sort_values(by='Importance', ascending=True)

    fig = px.bar(df_plot, x='Importance', y='Feature', orientation='h',
                 labels={'Importance': 'Feature Importance'})

    return html.Iframe(srcDoc=fig.to_html(), style={"width": "100%", "height": "470px"})


def get_client_feature_importance(client_id):
    expected_values, features, shap_values = fn.get_client_feature_importance(
        client_id)

    shap_df = pd.DataFrame({
        'feature': features,
        'shap_value': shap_values[1]
    })

    shap_df = shap_df.sort_values(by='shap_value', ascending=False)

    fig = px.bar(shap_df, x='shap_value', y='feature', orientation='h', labels={
                 'shap_value': 'SHAP Value', 'feature': 'Feature'})

    return html.Iframe(srcDoc=fig.to_html(), style={"width": "100%", "height": "470px"})


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

    return html.Iframe(srcDoc=fig.to_html(), style={"width": "100%", "height": "480px"})


def build_comparison_graphs(client_id, selected_features):
    client_data, neighbours_data = fn.get_client_neighbours(client_id)
    fig = make_subplots(rows=len(selected_features), cols=1,
                        subplot_titles=selected_features)

    for i, feature in enumerate(selected_features, start=1):
        client_value = client_data[feature].values[0]
        neighbours_values = neighbours_data[feature]
        counts = np.histogram(
            neighbours_values, bins='auto', density=True)[0].max()

        fig.add_shape(type="line", x0=client_value, y0=0, x1=client_value,
                      y1=counts, line=dict(color="#FFA88F",), row=i, col=1)

        fig.add_trace(go.Scatter(
            x=[client_value], y=[counts / 2],
            text=[f"Client Value: {client_value:.2f}"],
            mode="markers",
            marker=dict(color="#FFA88F", size=10, opacity=0),
            hoverinfo="text",
            showlegend=False
        ), row=i, col=1)

        fig.add_trace(go.Histogram(x=neighbours_values, histnorm='probability density',
                      name=f'{feature} - Neighbours'), row=i, col=1)
        fig.update_xaxes(title_text="Feature Value", row=i, col=1)
        fig.update_yaxes(title_text="Density", row=i, col=1)

    fig.update_layout(height=300 * len(selected_features), showlegend=False)
    return html.Iframe(srcDoc=fig.to_html(), style={"width": "100%", "height": "350px"})


############################################################################################
##################                       PANELS                         ####################
############################################################################################


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
                dbc.Col(dbc.Card(general, color="primary", outline=True)),
                dbc.Col(dbc.Card(financial, color="secondary", outline=True)),
                dbc.Col(dbc.Card(loan, color="info", outline=True),  width=2),
            ],
            className="mb-4",
        ),
    ]
    )


def build_local_feature_importance_panel(local_importance_graph):
    return dbc.Row(html.Div(local_importance_graph,
                            role='graph'), className="mb-4",)


def build_global_feature_importance_panel(global_importance):
    return dbc.Row(html.Div(global_importance, role="graph"), className="mb-4",)


def build_comparison_distribution_graph(features):
    return dbc.Container([
        dbc.Row(
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': feature, 'value': feature}
                         for feature in features],
                multi=True,
                value=['CREDIT_INCOME_PERCENT']
            )
        ),
        dbc.Row(html.Div(id='comparison-graphs')),
    ])


def build_mean_comparison_graph(comparison_graph):
    return dbc.Row(html.Div(comparison_graph))


############################################################################################
##################                         LAYOUT                       ####################
############################################################################################
app.layout = dbc.Container(
    [
        navbar,
        dbc.Row(dbc.Col(html.H1("Client Profile"), width=6, style={'margin-top': '40px', 'margin-bottom': '20px'},
                        className="offset-3 text-center"),),
        dbc.Row(
            dcc.Loading(
                id="loading-1",
                type="circle",
                children=html.Div(id='client_panel'),
            ),),
        dbc.Row(id='importance_title'),
        dbc.Row(id='local_importance_title'),
        dbc.Row(
            dcc.Loading(
                id="loading-2",
                type="circle",
                children=html.Div(id='local_feature_importance_graph'),

            ),),
        dbc.Row(id='global_importance'),
        dbc.Row(
            dcc.Loading(
                id="loading-3",
                type="circle",
                children=html.Div(id='global_feature_importance_graph'),

            ),),
        dbc.Row(id='comparison_title'),
        dbc.Row(id='distribution_title'),
        dbc.Row(
            dcc.Loading(
                id="loading-4",
                type="circle",
                children=html.Div(id='distribution_graph'),
            ),),
        dbc.Row(id='mean_title'),
        dbc.Row(
            dcc.Loading(
                id="loading-5",
                type="circle",
                children=html.Div(id='mean_graph'),))
    ],
    fluid=True
)


############################################################################################
##################                         CALLBACKS                       #################
############################################################################################

@app.callback(
    Output('importance_title', 'children'),
    Output('local_importance_title', 'children'),
    Output('global_importance', 'children'),
    Input('client-dropdown', 'value'),
)
def get_importance_titles(client_id):
    if client_id is None:
        return html.Div(), html.Div(), html.Div()
    importance = dbc.Col(html.H2("Feature Importance"), width=6, style={
        'margin-top': '40px', 'margin-bottom': '40px'}, className="offset-3 text-center")
    local_importance = dbc.Col(
        html.H4("Local Feature Importance Using SHAP Values"))
    global_importance = dbc.Col(html.H4("Global Feature Importance"))

    return importance, local_importance, global_importance


@app.callback(
    Output('comparison_title', 'children'),
    Output('distribution_title', 'children'),
    Output('mean_title', 'children'),
    Input('client-dropdown', 'value'),
)
def get_comparison_titles(client_id):
    if client_id is None:
        return html.Div(), html.Div(), html.Div()
    comparison = dbc.Col(html.H2("Client Comparison"), width=6, style={
                         'margin-top': '40px', 'margin-bottom': '40px'}, className="offset-3 text-center")
    distribution = html.H4(
        "Distribution comparison with 20 nearest neighbours that were granted a loan")

    mean = html.H4(
        "Comparison to mean values of 20 nearest neighbours that were granted a loan")

    return comparison, distribution, mean


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
    Output('local_feature_importance_graph', 'children'),
    Input('client-dropdown', 'value'),
)
def get_local_feature_importance(client_id):
    if client_id is None:
        return
    local_importance = get_client_feature_importance(client_id)

    return build_local_feature_importance_panel(local_importance)


@app.callback(
    Output('global_feature_importance_graph', 'children'),
    Input('client-dropdown', 'value'),
)
def get_global_feature_importance(client_id):
    global_importance = get_global_importance()
    if client_id is None:
        return
    return build_global_feature_importance_panel(global_importance)


@app.callback(
    Output('mean_graph', 'children'),
    Input('client-dropdown', 'value'),
)
def get_comparison_information(client_id):
    if client_id is None:
        return

    return build_mean_comparison_graph(bar_comparison(client_id))

@app.callback(
    Output('distribution_graph', 'children'),
    Input('client-dropdown', 'value'),
)
def get_comparison_information(client_id):
    if client_id is None:
        return

    return build_comparison_distribution_graph(fn.get_all_features())


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
    app.run_server(debug=False)

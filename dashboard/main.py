from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import functions as fn

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])


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
                    [html.Strong("Work Seniority:"),
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
                        [html.Strong("Score: "), f" {prediction['probability'][0]} ", dbc.Badge(
                            "Granted", color="success", className="me-1") if prediction['prediction'][0] == 0 else dbc.Badge("Denied", color="danger", className="me-1")],
                        className="card-text",
                    ),

                    ]
        )], color="info",  outline=True)

    return dbc.Row(
        [
            dbc.Col(dbc.Card(general, color="primary", outline=True)),
            dbc.Col(dbc.Card(financial, color="secondary", outline=True)),
            dbc.Col(dbc.Card(loan, color="info", outline=True)),
        ],
        className="mb-4",
    )


############################################################################################
##################                         LAYOUT                       ####################
############################################################################################

app.layout = dbc.Container(
    [
        html.H1(children='Prêt à Dépenser', style={'textAlign': 'center', 'margin-top': '20px'}),
        dcc.Dropdown(fn.get_clients(), placeholder='Select a client',
                         id='client-dropdown', style={'width': '50%', 'margin': 'left'}),
        html.Div(id='client_panel',  style={'margin-top': '20px'})
        # dbc.Row(
        #     [

        #         dbc.Col(html.Div(id='graph',
        #                 style={'margin-top': '20px'}), md=8),
        #     ],
        #     align="center",
        # ),
    ],
    fluid=True,
    className="theme-minty"
)


@app.callback(
    Output('client_panel', 'children'),
    Input('client-dropdown', 'value')
)
def get_client_info(client_id):
    if client_id is None:
        return
    client_info = fn.get_client_info(client_id)
    client_prediction = fn.get_prediction(client_id)

    return build_client_panel(client_info, client_prediction)


if __name__ == '__main__':
    app.run_server(debug=True)

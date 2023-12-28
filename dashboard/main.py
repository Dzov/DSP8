from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import functions as fn

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Prêt à Dépenser', style={'textAlign': 'center'}),
    dcc.Dropdown(fn.get_clients(), placeholder='Select a client',
                 id='client-dropdown'),
    html.Div(id='client-info-output')
])


@app.callback(
    Output('client-info-output', 'children'),
    Input('client-dropdown', 'value')
)
def get_client_info(client_id):
    print(client_id)
    # client_info = fn.get_client_info(client_id)
    client_prediction = fn.get_prediction(client_id)
    print(client_prediction)

    return html.Div([
        html.H1(f'Client Information: {client_id}'),
        html.Div([
            html.P(
                f"This client has been {'granted' if client_prediction['prediction'][0] == 0 else 'denied'} a loan."),
            html.P(f"Score: {client_prediction['probability'][0]}")
        ])
    ])


if __name__ == '__main__':
    app.run_server(debug=True)

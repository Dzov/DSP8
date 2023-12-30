import requests
import json
import numpy as np
import pandas as pd

# url = "https://pret-a-depenser.azurewebsites.net"
url = " http://127.0.0.1:8000"


def get_prediction(client_id):
    body = {"clientId": client_id}
    return requests.post(f"{url}/predict", json=body).json()


def get_clients():
    return requests.get(f"{url}/clients").json()


def get_client_info(client_id):
    return requests.get(f"{url}/clients/{client_id}").json()


def get_all_features():
    return requests.get(f"{url}/features").json()


def get_client_neighbours(client_id):
    response = requests.get(f"{url}/clients/{client_id}/neighbours").json()
    client_data = pd.DataFrame.from_dict(json.loads(response['client_data']))
    neighbours_data = pd.DataFrame.from_dict(
        json.loads(response['neighbours_data']))
    return client_data, neighbours_data


def get_client_feature_importance(client_id):
    response = requests.get(f"{url}/clients/{client_id}/shap").json()
    shap_values = json.loads(response["shap_values"])
    shap_values = {int(k): v for k, v in shap_values.items()}

    shap_force_plot_values = np.array(
        [shap_values[i][0] for i in range(len(shap_values))])
    return response['expected_values'], response['features'], shap_force_plot_values


def get_global_feature_importance():
    response = requests.get(f"{url}/shap").json()
    return pd.DataFrame.from_dict(json.loads(response["importance"]), orient='index')

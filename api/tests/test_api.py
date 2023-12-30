import pandas as pd
from pandas.testing import assert_frame_equal
import os
import json
import numpy as np
from fastapi.testclient import TestClient
from .. import main

client = TestClient(main.api)

data = pd.read_csv(os.path.join('data', 'client_profiles.csv'))
data['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)
data.drop(columns='Unnamed: 0', inplace=True)


def get_client_id():
    return int(data.loc[0, 'SK_ID_CURR'])


def test_home():
    response = client.get('/')
    assert response.status_code == 200


def test_clients():
    response = client.get('/clients')
    assert response.status_code == 200


def test_client():
    id = get_client_id()
    response = client.get(f'/clients/{id}')
    assert response.status_code == 200


def test_neighbours():
    id = get_client_id()
    response = client.get(f'/clients/{id}/neighbours')
    assert response.status_code == 200


def test_shap():
    response = client.get('/shap')
    assert response.status_code == 200


def test_features():
    response = client.get('/features')
    assert response.status_code == 200


def test_client_shap():
    id = get_client_id()
    response = client.get(f'/clients/{id}/shap')
    assert response.status_code == 200


def test_post_predict():
    client_id = get_client_id()
    response = client.post(
        "/predict/",
        json={"clientId": client_id},
    )
    assert response.status_code == 200
    assert response.json().get('clientId') == client_id


def test_client_not_found():
    response = client.post(
        "/predict/",
        json={"clientId": -1},
    )
    assert response.status_code == 404


def test_client_bad_request():
    clientId = get_client_id()
    response = client.post(
        "/predict/",
        json={"clientdId": clientId},
    )
    assert response.status_code == 422

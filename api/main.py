from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import os
import pandas as pd
import numpy as np
import shap
from sklearn.neighbors import NearestNeighbors

api = FastAPI()


class BodyItem(BaseModel):
    clientId: int


class ResponseItem(BaseModel):
    prediction: int
    probability: float
    predictionThreshold: float
    clientId: int


class Client(BaseModel):
    clientId: int
    childrenCount: int
    maritalStatus: str
    totalIncome: int
    age: int
    workSeniority: int
    occupation: str
    gender: str
    credit: int


def get_model():
    with open('model/model.pkl', 'rb') as file:
        return pickle.load(file)


def get_model_data(client_id: int = None):
    data_path = os.path.join('data', 'sample_client_data.csv')
    data = pd.read_csv(data_path)
    data.drop(columns='Unnamed: 0', inplace=True)

    if client_id:
        client_info = data[data.SK_ID_CURR == int(client_id)]
        if client_info.empty:
            raise HTTPException(
                status_code=404, detail=f"Client ID {client_id} not found.")
        return client_info

    return data


def get_historical_data():
    data_path = os.path.join('data', 'sample_historical_data.csv')
    data = pd.read_csv(data_path)
    data.drop(columns='Unnamed: 0', inplace=True)

    return data


def get_neighbours(client_id: int):
    data = get_historical_data()
    data = data[data['TARGET'] == 0].drop(columns='TARGET')
    x = data.values

    client_data = get_model_data(client_id)
    client_data = client_data[data.columns]

    neighbours = NearestNeighbors(n_neighbors=20)
    neighbours.fit(x)

    indexes = neighbours.kneighbors(
        X=client_data.values, n_neighbors=20, return_distance=False).ravel()

    neighbours_indexes = list(data.iloc[indexes].index)

    return data.loc[neighbours_indexes, :], client_data


def get_client_profile_data():
    data_path = os.path.join('data', 'client_profiles.csv')
    data = pd.read_csv(data_path)
    data.drop(columns='Unnamed: 0', inplace=True)
    return data


def build_client_response(client_id: int):
    data = get_client_profile_data()
    client = data[data.SK_ID_CURR == int(client_id)]
    if client.empty:
        raise HTTPException(
            status_code=404, detail=f"Client ID {client_id} not found.")

    return {
        "clientId": client_id,
        "childrenCount": client.CNT_CHILDREN.tolist()[0],
        "maritalStatus": client.NAME_FAMILY_STATUS.tolist()[0],
        "totalIncome": client.AMT_INCOME_TOTAL.tolist()[0],
        "age": round(client.DAYS_BIRTH.tolist()[0]/360),
        "workSeniority": round(client.DAYS_EMPLOYED.tolist()[0]/360),
        "occupation": client.OCCUPATION_TYPE.tolist()[0],
        "gender": client.CODE_GENDER.tolist()[0],
        "credit": client.AMT_CREDIT.tolist()[0],
    }


def predict_loan_eligibility(client_id: int):
    client_info = get_model_data(client_id)
    proba = get_model().predict_proba(client_info.values)[:, 1]
    prediction = (proba > 0.1).astype(int)
    return {
        "clientId": client_id,
        "probability": np.round(proba, 2).tolist(),
        "predictionThreshold": 0.1,
        "prediction": prediction.tolist(),
    }


def get_feature_importance():
    model = get_model().steps[-1][1]
    data = get_model_data()
    importance = pd.DataFrame(model.feature_importances_.reshape(
        1, -1), columns=data.columns).T.sort_values(by=0, ascending=False)
    return importance.head(25)

#################### ENDPOINTS ####################

@api.get('/')
async def home():
    return {'message': "Prêt à Dépenser"}


@api.post('/predict', response_model=ResponseItem)
async def predict(item: BodyItem):
    response_data = predict_loan_eligibility(item.clientId)
    return JSONResponse(content=response_data)


@api.get('/clients')
async def get_clients():
    client_ids = get_client_profile_data().SK_ID_CURR.to_list()
    return JSONResponse(content=client_ids)


@api.get('/clients/{id}', response_model=Client)
async def get_client(id: int):
    client = build_client_response(id)
    return JSONResponse(content=client)


@api.get('/clients/{id}/neighbours')
async def get_client_neighbours(id: int):
    neighbours, client_data = get_neighbours(id)
    response = {
        "client_data": client_data.to_json(),
        "neighbours_data": neighbours.to_json(),
    }
    return JSONResponse(content=response)


@api.get('/shap')
async def get_global_feature_importance():
    feature_importance = get_feature_importance()
    response = {
        "importance": feature_importance.to_json()
    }

    return JSONResponse(content=response)


@api.get('/features')
async def get_features():
    data = get_historical_data()
    data = data.drop(columns='TARGET')

    return JSONResponse(content=data.columns.tolist())


@api.get('/clients/{id}/shap')
async def get_client_feature_importance(id: int):
    data = get_model_data(id)
    explainer = shap.TreeExplainer(get_model().steps[-1][1])
    shap_values = pd.Series(list(explainer.shap_values(data.values)))
    expected_values = explainer.expected_value
    response = {
        "shap_values": shap_values.to_json(),
        "expected_values": expected_values,
        "features": data.columns.tolist()
    }

    return JSONResponse(content=response)

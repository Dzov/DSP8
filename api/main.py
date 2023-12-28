from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import os
import pandas as pd
import numpy as np

api = FastAPI()

data_path = os.path.join('api/data', 'displayable_client_info.csv')
data = pd.read_csv(data_path)
data.drop(columns='Unnamed: 0', inplace=True)
data['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)


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
    lgbm = open('api/model/model.pkl', 'rb')
    return pickle.load(lgbm)


def get_client_information(client_id:int):
    client_info = data[data.SK_ID_CURR == int(client_id)]
    if client_info.empty:
        raise HTTPException(
            status_code=404, detail=f"Client ID {client_id} not found.")

    return client_info

def build_client_response(client_id: int):
    try:
        client = get_client_information(client_id)
    except HTTPException as e:
        raise
    return {
        "clientId": client_id,
        "childrenCount": client.CNT_CHILDREN.tolist()[0],
        "maritalStatus": client.NAME_FAMILY_STATUS.tolist()[0],
        "totalIncome": int(client.AMT_INCOME_TOTAL.tolist()[0]),
        "age": np.round(-client.DAYS_BIRTH.tolist()[0]/360),
        "workSeniority": np.round(-client.DAYS_EMPLOYED.tolist()[0]/360),
        "occupation": client.OCCUPATION_TYPE.tolist()[0],
        "gender": client.CODE_GENDER.tolist()[0],
        "credit": int(client.AMT_CREDIT.tolist()[0]),
    }

def get_client_ids():
    return data.SK_ID_CURR.to_list()

def predict_loan_eligibility(client_id: int):
    data_path = os.path.join('api/data', 'sample_client_data.csv')
    data = pd.read_csv(data_path)
    data.drop(columns='Unnamed: 0', inplace=True)

    client_info = data[data.SK_ID_CURR == int(client_id)]
    if client_info.empty:
        raise HTTPException(
            status_code=404, detail=f"Client ID {client_id} not found.")
    
    proba = get_model().predict_proba(client_info.values)[:, 1]
    prediction = (proba > 0.1).astype(int)
    return {
        "clientId": client_id,
        "probability": np.round(proba, 2).tolist(),
        "predictionThreshold": 0.1,
        "prediction": prediction.tolist(),
    }


@api.get('/')
async def home():
    return {'message': "Prêt à Dépenser"}


@api.post('/predict', response_model=ResponseItem)
async def predict(item: BodyItem):
    response_data = predict_loan_eligibility(item.clientId)
    return JSONResponse(content=response_data)

@api.get('/clients')
async def get_clients():
    return JSONResponse(content=get_client_ids())

@api.get('/clients/{id}', response_model=Client)
async def get_client(id: int):
    client = build_client_response(id)
    print(client)
    return JSONResponse(content=client)

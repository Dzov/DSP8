
FROM python:3.11

WORKDIR /api

COPY ./requirements.txt /api/requirements.txt
COPY ./main.py /api/main.py 
COPY ./model/model.pkl /api/model/model.pkl 
COPY ./data/sample_client_data.csv /api/data/sample_client_data.csv

RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt

CMD [ "uvicorn", "main:api", "--host", "0.0.0.0", "--port", "80"]
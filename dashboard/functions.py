import requests

url = "https://pret-a-depenser.azurewebsites.net"

def get_prediction(client_id):
    print(f"hello{client_id}")
    body = {"clientId": client_id}
    response = requests.post(f"{url}/predict", json=body).json()
    print(response)

def get_clients():
    return requests.get(f"{url}/clients").json()

def get_client_info(client_id):
    return requests.get(f"{url}/clients/{client_id}").json()



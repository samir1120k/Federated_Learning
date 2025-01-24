import requests

def send_model_update(client_id, update):
    url = 'http://localhost:5000/update'
    payload = {'client_id': client_id, 'update': update}
    response = requests.post(url, json=payload)
    print(response.json())

def get_aggregated_model():
    url = 'http://localhost:5000/get_model'
    response = requests.get(url)
    print(response.json())

# Example usage:
client_id = 'client_1'
model_update = {'weights': [0.1, 0.2, 0.3]}  # Example update

# Send model update
send_model_update(client_id, model_update)

# Get aggregated model
get_aggregated_model()

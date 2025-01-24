from flask import Flask, request, jsonify
import json

app = Flask(__name__)

# Storing the aggregated model updates
model_updates = {}

@app.route('/update', methods=['POST'])
def update_model():
    # Receive client model updates
    client_data = request.get_json()
    client_id = client_data['client_id']
    update = client_data['update']
    
    # Store or aggregate the updates
    model_updates[client_id] = update
    
    return jsonify({"message": "Model update received."})

@app.route('/get_model', methods=['GET'])
def get_model():
    # Return the aggregated model updates
    return jsonify(model_updates)

if __name__ == '__main__':
    app.run(port=5000)

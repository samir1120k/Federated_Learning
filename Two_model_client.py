# Client-side
import socket
import torch
import io

def receive_model(conn):
    """
    Receives the model architecture and state_dict from the server.

    Args:
        conn: The socket connection.

    Returns:
        The received model.
    """
    try:
        # Receive model architecture
        data_len = int.from_bytes(conn.recv(4), byteorder='big')
        model_bytes = b''
        while len(model_bytes) < data_len:
            model_bytes += conn.recv(data_len - len(model_bytes))
        buffer = io.BytesIO(model_bytes)
        model = torch.load(buffer)

        # Receive state_dict
        data_len = int.from_bytes(conn.recv(4), byteorder='big')
        state_dict_bytes = b''
        while len(state_dict_bytes) < data_len:
            state_dict_bytes += conn.recv(data_len - len(state_dict_bytes))
        buffer = io.BytesIO(state_dict_bytes)
        state_dict = torch.load(buffer)
        model.load_state_dict(state_dict)

        return model

    except Exception as e:
        print(f"Error receiving model: {e}")
        return None

# Create a socket
host = '127.0.0.1'
port = 12345
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))

# Receive the model
received_model = receive_model(client_socket)

if received_model:
    print("Model received successfully.")
    # Use the received model 
    # ... (e.g., perform local training) ... 
    print(received_model) 

client_socket.close()
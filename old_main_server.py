import socket
import threading
import random
import os
import torch
import torch.nn as nn
from queue import Queue
import time

# CNN Model Definition
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 500)
        self.fc2 = nn.Linear(500, 16)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Global Variables
global_model = CNN()
clients = []
model_updates = Queue()
lock = threading.Lock()


def save_model_to_file(model, filename="global_model.pt"):
    torch.save(model.state_dict(), filename)


def load_model_from_file(model, filename):
    """
    Load model weights safely from the given file.
    """
    try:
        model.load_state_dict(torch.load(filename, weights_only=True))
        print(f"Model weights loaded successfully from {filename}.")
    except Exception as e:
        print(f"Error loading model weights: {e}")


def send_model(client_socket):
    try:
        filename = "global_model.pt"
        save_model_to_file(global_model, filename)
        filesize = os.path.getsize(filename)

        # Send metadata
        metadata = f"{filename}\n{filesize}".encode()
        client_socket.sendall(metadata)

        # Send model file in chunks
        with open(filename, "rb") as f:
            while chunk := f.read(1024):
                client_socket.sendall(chunk)

        print("Model sent successfully.")

    except Exception as e:
        print(f"Error sending model: {e}")


def receive_model(client_socket):
    try:
        # Receive metadata
        metadata = client_socket.recv(1024).decode().strip().split("\n")
        filename, filesize = metadata[0], int(metadata[1])

        # Receive the file
        with open(filename, "wb") as f:
            received = 0
            while received < filesize:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)

        # Load the model
        client_model = CNN()
        load_model_from_file(client_model, filename)
        os.remove(filename)  # Remove temporary file
        return client_model.state_dict()

    except Exception as e:
        print(f"Error receiving model: {e}")
        return None


def handle_client(client_socket, client_address):
    with lock:
        clients.append(client_socket)

    print(f"Client {client_address} connected.")

    try:
        while True:
            data = client_socket.recv(1024).decode()
            if data == "REQUEST_MODEL":
                send_model(client_socket)
            elif data == "SEND_UPDATE":
                model_state = receive_model(client_socket)
                if model_state:
                    with lock:
                        model_updates.put(model_state)
    except Exception as e:
        print(f"Client {client_address} disconnected: {e}")
    finally:
        with lock:
            if client_socket in clients:
                clients.remove(client_socket)
        client_socket.close()


def aggregate_models(global_model, model_updates, num_updates):
    print("Aggregating models...")
    aggregated_state = global_model.state_dict()
    updates = [model_updates.get() for _ in range(num_updates)]  # Retrieve updates from the queue
    for key in aggregated_state.keys():
        aggregated_state[key] = torch.stack([update[key] for update in updates]).mean(dim=0)
    global_model.load_state_dict(aggregated_state)
    print("Global model updated successfully.")

start_time = time.time()
end_time = start_time + 30

def main():
    host = "127.0.0.1"
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(10)
    print("Server is running...")

    try:
        while True:
            if len(clients) < 2:
                client_socket, client_address = server_socket.accept()
                threading.Thread(target=handle_client, args=(client_socket, client_address)).start()
            elif len(clients) >= 2:
                selected_clients = random.sample(clients, 2)

                # Request models from selected clients
                for client_socket in selected_clients:
                    client_socket.sendall("REQUEST_MODEL".encode())

                # Wait for model updates
                for _ in range(2):
                    model_state = receive_model(client_socket)
                    if model_state:
                        model_updates.put(model_state)

                # Aggregate models
                aggregate_models(global_model, model_updates, 2)

    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server_socket.close()


if __name__ == "__main__":
    main()

import socket
import threading
import random
import os
import torch
import torch.nn as nn
from queue import Queue
import time
import select

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
clients = {}
model_updates = Queue()
lock = threading.Lock()

def save_model_to_file(model, filename="global_model.pt"):
    torch.save(model.state_dict(), filename)

def load_model_from_file(model, filename):
    try:
        model.load_state_dict(torch.load(filename, weights_only=True))
        # print(f"Model weights loaded successfully from {filename}.")
    except Exception as e:
        print(f"Error loading model weights: {e}")

def send_model(client_socket):  # Function to send the global model to a client
    try:
        client_socket.setblocking(True)  # Set blocking mode
        filename = "global_model.pt"
        save_model_to_file(global_model, filename)
        filesize = os.path.getsize(filename)

        # Send metadata with a delimiter
        metadata = f"{filename}\n{filesize}\n".encode()
        client_socket.sendall(metadata)

        # Send model file in chunks
        with open(filename, "rb") as f:
            while chunk := f.read(1024):
                client_socket.sendall(chunk)

        print("Model sent successfully to:", client_socket.getpeername())
    except Exception as e:
        print(f"Error sending model: {e}")


def receive_model(client_socket):
    try:
        client_socket.setblocking(True)  # Set blocking mode

        # Receive metadata with a fixed buffer and retry if incomplete
        metadata = b""
        while b"\n" not in metadata:
            chunk = client_socket.recv(1024)
            if not chunk:
                print("Client disconnected unexpectedly. Incomplete metadata received.")
                return None
            metadata += chunk

        metadata = metadata.decode().strip().split("\n")
        if len(metadata) < 2:
            print("Error: Incomplete metadata received.")
            return None

        filename, filesize = metadata[0], int(metadata[1])

        # Receive the file
        with open(filename, "wb") as f:
            received = 0
            while received < filesize:
                chunk = client_socket.recv(1024)
                if not chunk:
                    print("Client disconnected unexpectedly. Incomplete model received.")
                    return None
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

def handle_client(client_socket, client_address):  # Function to handle client connections
    with lock:
        clients[client_address] = {
            'socket': client_socket,
            'connected_at': time.ctime(),
        }

def aggregate_models(global_model, model_updates, num_updates):  # Function to aggregate models
    print("Aggregating models...")
    aggregated_state = global_model.state_dict()
    updates = [model_updates.get() for _ in range(num_updates)]  # Retrieve updates from the queue
    if len(updates) < num_updates:
        print("Warning: Fewer updates received than expected.")
        return global_model  # Return the original model if not enough updates

    for key in aggregated_state.keys():
        aggregated_state[key] = torch.stack([update[key] for update in updates]).mean(dim=0)

    global_model.load_state_dict(aggregated_state)
    print("Global model updated successfully.")
    return global_model

def select_and_notify_clients():
    if len(clients) < 2:
        print("Not enough clients connected for selection.")
        return

    selected_clients = random.sample(list(clients.values()), 2)
    for client in selected_clients:
        client_socket = client['socket']
        send_model(client_socket)

        model_state = receive_model(client_socket)  # Receive model from the client
        if model_state:
            print(f"Model received from {client_socket.getpeername()}.")
            model_updates.put(model_state)  # Store the received model state

import select

def main():
    host = "127.0.0.1"
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(10)
    server_socket.setblocking(False)  # Set the server socket to non-blocking mode
    print("Server is running...")

    try:
        for i in range(2):
            while True:
                start = time.time()
                end = time.time() + 30  # Run for 30 seconds

                while time.time() <= end:
                    try:
                        # Use select to wait for a client connection with a timeout of 1 second
                        readable, _, _ = select.select([server_socket], [], [], 1)
                        
                        if readable:
                            client_socket, client_address = server_socket.accept()
                            print(f"Accepted connection from {client_address}")
                            client_handler = threading.Thread(target=handle_client, args=(client_socket, client_address))
                            client_handler.start()

                    except select.error as e:
                        # Handle errors with select or socket
                        print(f"Error: {e}")
                        continue

                # After receiving 2 clients, select and notify
                if len(clients) >= 2:
                    select_and_notify_clients()
                    aggregate_models(global_model, model_updates, 2)
                else:
                    print("Not enough clients connected to proceed.")
                break  # Stop accepting new clients after selection

    finally:
        server_socket.close()


if __name__ == "__main__":
    main()

import socket
import threading
import random
import time

# Dictionary to store client details
clients = {}

def handle_client(client_socket, client_address):

    try:
        # Store client details in the dictionary
        clients[client_address] = {
            'socket': client_socket,
            'connected_at': time.ctime(),
            'messages': []
        }

        while True:
            data = client_socket.recv(1024)
            if not data:
                break

            message = data.decode('utf-8')
            print(f"Received message from {client_address}: {message}")
            clients[client_address]['messages'].append(message)
            response = f"Server received your message: {message}"
            client_socket.send(response.encode('utf-8'))

    except Exception as e:
        print(f"Error handling client {client_address}: {e}")
    finally:
        # Clean up
        client_socket.close()
        del clients[client_address]

def select_and_notify_clients():

    if len(clients) < 2:
        print("Not enough clients connected for selection.")
        return

    selected_clients = random.sample(list(clients.values()), 2)
    for client in selected_clients:
        client_socket = client['socket']
        notification = "You are selected for training."
        try:
            client_socket.send(notification.encode('utf-8'))
            print(f"Sent notification to {client_socket.getpeername()}: {notification}")
        except Exception as e:
            print(f"Error sending notification to {client_socket.getpeername()}: {e}")


def main():
    """
    Starts the server and handles client connections.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '127.0.0.1'
    port = 12345
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Accepted connection from {client_address}")
            client_handler = threading.Thread(target=handle_client, args=(client_socket, client_address))
            client_handler.start()

            # After receiving 10 clients, select and notify
            if len(clients) == 3:
                select_and_notify_clients()
                break  # Stop accepting new clients after selection

    finally:
        server_socket.close()

if __name__ == "__main__":
    main()

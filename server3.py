import socket
import threading
import time
import pandas as pd

# Dictionary to store active client details
clients = {}
# List to store finalized client data
client_data = []
client_threads = []

def handle_client(client_socket, client_address):
    """
    Handles communication with a single client.

    Args:
        client_socket: The socket object for communication with the client.
        client_address: The address of the client.
    """
    try:
        # Store client details in the dictionary
        clients[client_address] = {
            'connected_at': time.time(),
            'port': client_address[1],
            'address': client_address[0],
            'messages': []
        }

        # Calculate the end time for this client's handling
        end_time = time.time() + 60  # Handle client for 60 seconds

        while time.time() < end_time:
            client_socket.settimeout(max(0, end_time - time.time()))  # Adjust timeout to the remaining time
            try:
                data = client_socket.recv(1024)
                if not data:
                    break

                message = data.decode('utf-8')
                print(f"Received message from {client_address}: {message}")
                clients[client_address]['messages'].append(message)
                response = "Server received your message: " + message
                client_socket.send(response.encode('utf-8'))
            except socket.timeout:
                break

    finally:
        # Store finalized client details in client_data
        if client_address in clients:
            client_info = clients.pop(client_address)
            for message in client_info['messages']:
                client_data.append({
                    'Address': client_info['address'],
                    'Port': client_info['port'],
                    'Connected At': time.ctime(client_info['connected_at']),
                    'Message': message
                })
        client_socket.close()

def main():
    """
    Starts the server and handles client connections.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '127.0.0.1'
    port = 12345
    server_socket.bind((host, port))
    server_socket.listen(5)
    server_socket.settimeout(1)  # Set a timeout for accept()
    print(f"Server listening on {host}:{port}")

    start_time = time.time()

    try:
        while True:
            # Check if the server has exceeded its runtime limit
            if time.time() - start_time >= 60:
                print("Server runtime exceeded 1 minute. Shutting down...")
                break

            try:
                client_socket, client_address = server_socket.accept()
                print(f"Accepted connection from {client_address}")
                client_handler = threading.Thread(target=handle_client, args=(client_socket, client_address))
                client_handler.start()
                client_threads.append(client_handler)
            except socket.timeout:
                # Continue the loop to check the time limit
                continue

    finally:
        # Wait for all client threads to finish
        for thread in client_threads:
            thread.join()

        # Create a DataFrame from the finalized client data
        df = pd.DataFrame(client_data)
        print("\nClient Details DataFrame:")
        print(df)

        server_socket.close()
        return df

if __name__ == "__main__":
    df = main()

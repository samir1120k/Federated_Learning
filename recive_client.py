import socket

def client_program():
    host = '127.0.0.1'  # Server's IP address
    port = 12345        # Server's port

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    try:
        while True:
            # Listen for messages from the server
            response = client_socket.recv(1024).decode('utf-8')
            print(f"Server response: {response}")

            # Allow the client to send messages
            message = input("Enter a message for the server (or 'exit' to quit): ")
            if message.lower() == 'exit':
                break

            client_socket.send(message.encode('utf-8'))

    finally:
        client_socket.close()

if __name__ == "__main__":
    client_program()

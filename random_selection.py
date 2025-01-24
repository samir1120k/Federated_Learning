import random

def select_clients_for_training(client_data):
    """
    Randomly selects 2 clients from the given client data and sends a message.

    Args:
        client_data (list): List of dictionaries containing client details.

    Returns:
        list: List of selected client details.
    """
    if len(client_data) < 2:
        print("Not enough clients to select for training.")
        return []

    # Randomly select 2 clients
    selected_clients = random.sample(client_data, 2)

    # Notify selected clients
    for client in selected_clients:
        print(f"Sending message to {client['Address']}:{client['Port']}")
        print(f"Message: 'You are selected for training.'")
        # Add more logic here if you want to send the message via socket

    return selected_clients

# Example usage
client_data = [
    {'Address': '127.0.0.1', 'Port': 12345, 'Connected At': '2025-01-21 10:30:45', 'Message': 'Hello Server'},
    {'Address': '127.0.0.2', 'Port': 12346, 'Connected At': '2025-01-21 10:31:45', 'Message': 'Hi'},
    {'Address': '127.0.0.3', 'Port': 12347, 'Connected At': '2025-01-21 10:32:45', 'Message': 'Training Info'},
    {'Address': '127.0.0.4', 'Port': 12348, 'Connected At': '2025-01-21 10:33:45', 'Message': 'Ping'},
    {'Address': '127.0.0.5', 'Port': 12349, 'Connected At': '2025-01-21 10:34:45', 'Message': 'Hello'},
    {'Address': '127.0.0.6', 'Port': 12350, 'Connected At': '2025-01-21 10:35:45', 'Message': 'Help'},
    {'Address': '127.0.0.7', 'Port': 12351, 'Connected At': '2025-01-21 10:36:45', 'Message': 'Testing'},
    {'Address': '127.0.0.8', 'Port': 12352, 'Connected At': '2025-01-21 10:37:45', 'Message': 'Client'},
    {'Address': '127.0.0.9', 'Port': 12353, 'Connected At': '2025-01-21 10:38:45', 'Message': 'Check'},
    {'Address': '127.0.0.10', 'Port': 12354, 'Connected At': '2025-01-21 10:39:45', 'Message': 'Demo'}
]

selected_clients = select_clients_for_training(client_data)

print("\nSelected Clients for Training:")
for client in selected_clients:
    print(client)

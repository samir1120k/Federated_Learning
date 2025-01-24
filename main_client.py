import socket
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd

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


def receive_model(client_socket,model):
    """
    Receive the global model from the server.
    """
    try:
        # Receive metadata
        metadata = client_socket.recv(1024).decode().strip().split("\n")
        filename, filesize = metadata[0], int(metadata[1])

        # Receive the file
        with open(filename, "wb") as f:
            received = 0
            while received < filesize:
                data = client_socket.recv(1024)
                if not data:
                    break
                f.write(data)
                received += len(data)

        print("Model received successfully.")

        # Load the model safely
        
        model.load_state_dict(torch.load(filename, weights_only=True))
        os.remove(filename)  # Remove the temporary file
        return model

    except Exception as e:
        print(f"Error receiving model: {e}")
        return None



def send_model(client_socket, model):
    """
    Send the local model to the server.
    """
    try:
        filename = "client_model.pt"
        torch.save(model.state_dict(), filename)
        filesize = os.path.getsize(filename)

        # Send metadata
        client_socket.sendall(f"{filename}\n{filesize}".encode())

        # Send model file
        with open(filename, "rb") as f:
            while chunk := f.read(1024):
                client_socket.sendall(chunk)

        print("Model sent successfully.")
        os.remove(filename)  # Remove the temporary file

    except Exception as e:
        print(f"Error sending model: {e}")


train_data=pd.read_csv(r'optimized_flattened_dataset_batch_1.csv')


def train_model(model, train_data, optimizer, criterion, num_epochs=10):
    # Prepare data
    X_train = torch.tensor(train_data.iloc[:, 1:].values, dtype=torch.float32)
    y_train = torch.tensor(train_data.iloc[:, 0].values, dtype=torch.long)

        # Reshape data for convolutional layers (assuming 28x28 images)
    X_train = X_train.view(-1, 1, 28, 28)  # Reshape to (batch_size, channels, height, width)

    # Create a DataLoader for efficient batching
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size=32, shuffle=True
    )


    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad() 

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

def main():
    host = "127.0.0.1"  # Server IP address
    port = 12345        # Server port
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Initialize local model
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    try:
        while True:
            # Request global model from server
            client_socket.sendall("REQUEST_MODEL".encode())
            model = receive_model(client_socket,model)

            if model is None:
                print("Failed to receive model. Exiting...")
                break

            # Simulate local training
            model = train_model(model, train_data, optimizer, criterion)


            # Send updated model back to server
            client_socket.sendall("SEND_UPDATE".encode())
            send_model(client_socket, model)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        client_socket.close()
        print("Client connection closed.")


if __name__ == "__main__":
    main()

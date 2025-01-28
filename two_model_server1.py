import torch
import socket
import torch.nn as nn

class CNN1(nn.Module):
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
global_model1 = CNN1()

# Server-side
import socket
import torch
import io

def send_model(conn, model):
    """
    Sends the model architecture and state_dict to the client.

    Args:
        conn: The socket connection.
        model: The PyTorch model.
    """
    try:
        # Serialize model architecture
        buffer = io.BytesIO()
        torch.save(model, buffer)
        model_bytes = buffer.getvalue() 
        conn.sendall(len(model_bytes).to_bytes(4, byteorder='big')) 
        conn.sendall(model_bytes) 

        # Serialize state_dict
        buffer = io.BytesIO()
        
        
        torch.save(model.state_dict(), buffer)
        state_dict_bytes = buffer.getvalue()
        conn.sendall(len(state_dict_bytes).to_bytes(4, byteorder='big'))
        conn.sendall(state_dict_bytes)

    except Exception as e:
        print(f"Error sending model: {e}")

# ... (Your model definition and training) ... 

# Create a socket
host = '127.0.0.1'
port = 12345
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen()

while True:
    conn, addr = server_socket.accept()
    print('Connected by', addr)
    
    # Send the model
    send_model(conn, global_model1) 

    conn.close()
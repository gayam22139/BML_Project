import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module): 
    """
    basic multi-layer perceptron
    """
    def __init__(self, hidden_size=400):
        super().__init__()
        self.flat = Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, input):
        x = self.flat(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)  #(32,28,28) -> (32,784)
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) # (N,1,28,28) -> (N,32,28,28)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # (N,32,28,28) -> (N,32,14,14)

        self.fc1 = nn.Linear(32 * 14 * 14, 100)  
        self.fc2 = nn.Linear(100, 10) 

    def forward(self, x):
        # print(f"x input {x.shape}")
        x = self.pool(F.relu(self.conv1(x)))  # Convolutional layer followed by ReLU and max pooling
        # print(f"x before flattened {x.shape}")
        x = torch.flatten(x,start_dim=1,end_dim=-1) 
        # print(f"x after flattened {x.shape}")
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x #,F.softmax(x, dim=-1)  # Apply softmax to the output 
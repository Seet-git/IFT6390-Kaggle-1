import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_layer=256):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_layer, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLP(nn.Module):

    def __init__(self, input_size: int, hidden_size):
        super().__init__()

        # Layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Hyper-parameters
        self.criterion = None
        self.optimizer = None

    def forward(self, x_input) -> torch.Tensor:
        x_hidden = self.input_layer(x_input)
        activation_hidden = self.relu(x_hidden)
        x_output = self.hidden_layer(activation_hidden)
        return x_output


class Perceptron(nn.Module):

    def __init__(self, input_size: int):
        super().__init__()

        # Layers
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Hyper-parameters
        self.criterion = None
        self.optimizer = None

    def forward(self, x_input) -> torch.Tensor:
        x_input = self.linear(x_input)
        x_input = self.sigmoid(x_input)
        return x_input

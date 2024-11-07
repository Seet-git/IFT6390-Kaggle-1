import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns


class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scale_top_relu_ouput = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scale_bottom_relu_ouput = bottom_relu_output * self.w11

        input_to_final_relu = scale_bottom_relu_ouput + scale_top_relu_ouput + self.final_bias
        output = F.relu(input_to_final_relu)
        return output


class BasicNN_start(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 2)
        self.output_layer = nn.Linear(2, 1, bias=True)

    def forward(self, x_input):
        paths_output = F.relu(self.fc(x_input))
        output = F.relu(self.output_layer(paths_output))
        return output


def train_model(inputs, labels):
    model = BasicNN_start()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        total_loss = 0

        # Loop on all inputs
        for i in range(len(inputs)):
            # Predict = w * inputs[i] + b
            predict = model(inputs[i])

            # Loss = (y - predict)^2
            loss = (predict - labels[i]) ** 2

            total_loss += loss

            # Compute dL/dw et dL/db pour chaque poids w et biais b
            loss.backward()

        # w = w - lr * dL/dw et b = b - lr * dL/db
        optimizer.step()

        # Met à zéro les valeurs de dL/dw et dL/db pour tous les paramètres
        optimizer.zero_grad()

        # Affichage de la perte totale pour l'époque
        print(f"Époque {epoch + 1}, Perte totale : {total_loss.item()}")

        # Affichage des poids après la mise à jour
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Poids de {name} après mise à jour : {param.data}")


inputs = torch.tensor([0, 0.5, 1.])
labels = torch.tensor([0., 1., 0.])
train_model(inputs, labels)

inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
test_documents = np.load('../../data/data_test.npy', allow_pickle=True)

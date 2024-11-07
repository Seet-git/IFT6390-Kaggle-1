import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import random


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_res(history_tab, plot_type, x_label, y_label, title, figure_name):
    # Plot settings
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=17)
    plt.ylabel(y_label, fontsize=15)
    plt.xlabel(x_label, fontsize=15)

    # Plot results
    for i in history_tab:
        plt.plot(i[plot_type])

    # Save and show the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.savefig(f"./output/{figure_name}.eps", format='eps', bbox_inches='tight')
    plt.show()
    print(f"{figure_name}.esp saved")


class Perceptron(nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_input) -> torch.Tensor:
        x_input = self.linear(x_input)
        x_input = self.sigmoid(x_input)
        return x_input


def infer(model: Perceptron, data):
    # Évaluation du modèle
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in data:
            # Compute the prediction
            outputs = model(inputs)

            # Compute prediction
            pred = (outputs.squeeze() > 0.5).float()

            # Add prediction
            y_pred.append(pred.item())
            y_true.append(labels.item())

    # Calcul du F1-score
    return f1_score(y_true, y_pred)


def k_cross_validation(x_inputs: torch.Tensor, y_labels: torch.Tensor, k=5, epochs=50) -> None:
    # Split data
    k_fold = KFold(n_splits=k, shuffle=True)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Iterate k times
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(x_inputs)):
        #  Training set
        x_train = x_inputs[train_idx]
        y_train = y_labels[train_idx]

        # Validation set
        x_val = x_inputs[val_idx]
        y_val = y_labels[val_idx]

        # Set model
        model = Perceptron(x_train.shape[1], 1)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001)

        # Création du DataLoader
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset)

        # Création du DataLoader
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset)

        # Loop on all epochs
        for epoch in range(epochs):
            # Shuffle train set
            train_shuffle = np.arange(x_train.shape[0])
            np.random.shuffle(train_shuffle)

            train_loss = 0.0

            # Batch gradient descent
            for i in train_shuffle:
                # Model prediction
                pred = model.forward(x_train[i])

                # Compute loss
                loss = criterion(pred.squeeze(), y_train[i])
                train_loss += loss.item()

                # Backpropagation
                loss.backward()

                # Change weights
                optimizer.step()

                # Reset backward
                optimizer.zero_grad()

            f1_score_train = infer(model, train_loader)
            print(f'Train - Epoch [{epoch + 1}/{epochs}], F1 score: {f1_score_train:.4f}, Loss: {train_loss / len(x_train):.4f}')

            f1_score_val = infer(model, val_loader)
            print(f'Validation - Epoch [{epoch + 1}/{epochs}], F1 score: {f1_score_val:.4f} \n')

        f1_score_train = infer(model, train_loader)
        print(f'Train - Fold [{fold + 1}/{k}], F1 score: {f1_score_train:.4f}')

        f1_score_val = infer(model, val_loader)
        print(f'Validation - Fold [{fold + 1}/{k}], F1 score: {f1_score_val:.4f} \n')

    # plot_res([history], "train_loss", "K-folds", "Loss", "Loss train set", "plot_train_loss")


def main():
    set_seed(1)

    # Load data
    inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
    labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]

    # Set tensor
    x_tensor = torch.tensor(inputs_documents, dtype=torch.float32)
    y_tensor = torch.tensor(labels_documents, dtype=torch.float32)

    k_cross_validation(x_tensor, y_tensor)


if __name__ == '__main__':
    main()

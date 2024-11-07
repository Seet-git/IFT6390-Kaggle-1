import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Parameters
batch_size = 128
epochs = 10

# Load and process data
X_train =  np.load('../../data/data_train.npy', allow_pickle=True)
Y_train = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]


def compute_f1_score_macro(y_true, y_pred):
    """
    Calcule le F1 score macro.
    :param y_true: Labels réels
    :param y_pred: Prédictions du modèle
    :return: F1 score macro
    """
    # Récupérer les labels uniques
    labels = np.unique(y_true)

    # Initialiser les compteurs
    f1_scores = []

    for label in labels:
        # Vrais positifs
        vrai_pos = np.sum((y_true == label) & (y_pred == label))

        # Faux positifs
        faux_pos = np.sum((y_true != label) & (y_pred == label))

        # Faux négatifs
        faux_neg = np.sum((y_true == label) & (y_pred != label))

        # Accuracy et recall
        accuracy = vrai_pos / (vrai_pos + faux_pos) if (vrai_pos + faux_pos) > 0 else 0
        recall = vrai_pos / (vrai_pos + faux_neg) if (vrai_pos + faux_neg) > 0 else 0

        # F1 score
        f1 = (2 * accuracy * recall) / (accuracy + recall) if (accuracy + recall) > 0 else 0
        f1_scores.append(f1)

    # F1 score macro
    f1_macro = np.mean(f1_scores)
    return f1_macro


def split_dataset(inputs_train: np.array, labels_train: np.array):
    """
    Split the dataset into training and test set
    :param inputs_train:
    :param labels_train:
    :return:
    """
    # Shuffle the dataset
    indices = np.arange(len(inputs_train))
    np.random.shuffle(indices)

    inputs_shuffled = inputs_train[indices]
    labels_shuffled = labels_train[indices]

    # Get the size of the training set
    # train_size = int(np.ceil(0.8 * len(inputs_train)))
    train_size = len(inputs_train) - 2355

    # Train set
    set_train_inputs = inputs_shuffled[0:train_size, :]
    set_train_labels = labels_shuffled[0:train_size]

    # Test set
    set_test_inputs = inputs_shuffled[train_size:, :]
    set_test_labels = labels_shuffled[train_size:]

    return set_train_inputs, set_train_labels, set_test_inputs, set_test_labels

# Dataset class for PyTorch
class TextDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        else:
            return self.sequences[idx]

# DataLoaders
X_train, Y_train, X_test, Y_test = split_dataset(X_train, Y_train)
input_size = X_train.shape[1]
train_dataset = TextDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TextDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Define a simple feedforward model
class MLPClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Initialize model, loss function, and optimizer
model = MLPClassifier(input_size=input_size)
criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters())

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for sequences, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluation on the training set
model.eval()
test_loss, correct, total = 0.0, 0, 0
f1_scores = []
output_tab = []
with torch.no_grad():
    for sequences, labels in test_loader:
        outputs = model(sequences).squeeze()
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        f1_scores.extend(predicted)
        output_tab.extend(labels)


print(f'Training Loss: {test_loss/len(test_loader):.4f}, Accuracy: {compute_f1_score_macro(f1_scores, output_tab):.4f}')

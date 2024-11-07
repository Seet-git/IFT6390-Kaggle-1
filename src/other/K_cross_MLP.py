import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import KFold

# Charger les données
X_train = np.load('../../data/data_train.npy', allow_pickle=True)
y_train = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]

# Configuration des tenseurs
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Définition du modèle de réseau de neurones
class AdvancedNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.sigmoid(x)

# Paramètres du modèle
input_size = X_train.shape[1]
hidden_size = 128
output_size = 1
num_epochs = 100
batch_size = 32
learning_rate = 0.0001
k_folds = 5

# Fonction de validation croisée k-fold
def cross_validate_kfold(X, y, model_class, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold + 1}/{k}")

        # Préparer les sous-ensembles d'entraînement et de validation
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        # Créer les DataLoaders pour les sous-ensembles
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialiser le modèle, le critère et l'optimiseur
        model = model_class(input_size, hidden_size, output_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Entraînement
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()

        # Évaluation sur le fold de validation
        model.eval()
        y_pred = []
        y_true = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                predicted = (outputs.squeeze() > 0.5).float()
                y_pred.extend(predicted.numpy())
                y_true.extend(labels.numpy())

        # Calcul du score F1 pour ce fold
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
        print(f"Fold {fold + 1} F1 Score: {f1:.4f}")

    # Moyenne des scores F1
    mean_f1_score = np.mean(f1_scores)
    print(f"Mean F1 Score across all folds: {mean_f1_score:.4f}")
    return mean_f1_score

# Lancer la validation croisée
cross_validate_kfold(X_train_tensor, y_train_tensor, AdvancedNN, k=k_folds)

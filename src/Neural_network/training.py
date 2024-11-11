from types import SimpleNamespace

import torch
import wandb
import torch.nn as nn
from torch import optim
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
import src.config as config
from src.Neural_network.models import MLP_v2
from src.preprocessing import remove_low_high_frequency


def compute_loss(model, data):
    total_loss = 0.0
    for inputs, labels in data:
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        pred = model(inputs)
        loss = model.criterion(pred.view(-1), labels.view(-1))
        total_loss += loss.item()
    return total_loss / len(data)


def infer(model, data, threshold=0.5):
    # Initialisation
    model.eval()
    y_pred = []
    y_true = []

    # Compute prediction
    with torch.no_grad():
        for inputs, labels in data:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            pred = (torch.sigmoid(outputs).view(-1) > threshold).float()
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Compute f1 score, precision, recall, accuracy
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return f1, precision, recall, accuracy


def fit(model, train_loader, val_loader, epochs, infer_threshold, scheduler=None):
    """

    :param model:
    :param train_loader:
    :param val_loader:
    :param epochs:
    :param infer_threshold:
    :param scheduler:
    :return:
    """
    # Initialisation
    model.to(config.DEVICE)
    best_f1 = 0
    patience = 10
    patience_counter = 0

    # Loop on all epochs
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            model.optimizer.zero_grad()
            pred = model(inputs)
            loss = model.criterion(pred.view(-1), labels.view(-1))
            loss.backward()
            model.optimizer.step()

        # Compute results
        train_loss = compute_loss(model, train_loader)
        val_loss = compute_loss(model, val_loader)
        f1_train, _, _, _ = infer(model, train_loader, infer_threshold)
        f1_val, _, _, _ = infer(model, val_loader, infer_threshold)

        # Update learning rate
        if scheduler:
            scheduler.step(f1_val)

        if config.WANDB_ACTIVATE:
            wandb.log({
                "Epoch": epoch + 1,
                "Train loss": train_loss,
                "Validation loss": val_loss,
                "Train F1-score": f1_train,
                "Validation F1-score": f1_val,
            })

        # Prevent early stopping
        if f1_val > best_f1:
            best_f1 = f1_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def evaluation(hp, save_file=False, n_splits=2):
    """
    :param hp: 
    :param save_file:
    :param n_splits:
    :return:
    """
    # Convertit le dictionnaire en SimpleNamespace si nécessaire
    if isinstance(hp, dict):
        hp = SimpleNamespace(**hp)

    # Prétraitement des données
    X_train, X_test = remove_low_high_frequency(hp.low_frequency, hp.high_frequency)

    # K-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    f1_scores = []

    model = None
    # For all k folds
    for train_index, val_index in kf.split(X_train):
        # Inputs and labels
        inputs_train, inputs_val = X_train[train_index], X_train[val_index]
        labels_train, labels_val = config.LABELS_DOCUMENTS[train_index], config.LABELS_DOCUMENTS[val_index]

        # Set model
        model = MLP_v2(X_train.shape[1], hp.hidden_layer1, hp.hidden_layer2, hp.dropout_rate).to(config.DEVICE)

        # Compute weight difference to balance
        class_0_count = (config.LABELS_DOCUMENTS == 0).sum()
        class_1_count = (config.LABELS_DOCUMENTS == 1).sum()
        weight = torch.tensor([class_0_count / class_1_count - hp.minority_weight], dtype=torch.float32).to(
            config.DEVICE)
        weight = torch.clamp(weight, min=1.0)

        # Models parameters
        model.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        model.optimizer = optim.RMSprop(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, mode='max', factor=0.1, patience=3)

        # Define train set and val set
        train_dataset = TensorDataset(torch.tensor(inputs_train, dtype=torch.float32),
                                      torch.tensor(labels_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)
        val_dataset = TensorDataset(torch.tensor(inputs_val, dtype=torch.float32),
                                    torch.tensor(labels_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=False)

        # Train model
        fit(model, train_loader, val_loader, hp.epochs, hp.infer_threshold, scheduler)

        # Predict on validation set
        f1_val, _, _, _ = infer(model, val_loader, hp.infer_threshold)
        f1_scores.append(f1_val)

    if config.WANDB_ACTIVATE:
        wandb.finish()

    if save_file:
        return model, X_test, np.mean(f1_scores)

    return np.mean(f1_scores)
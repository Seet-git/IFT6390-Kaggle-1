import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import wandb

from src.NN.models import *
from src.NN.utils import *


def compute_loss(model, data):
    total_loss = 0.0
    # Mini-Batch
    for inputs, labels in data:
        # Model prediction
        pred = model.forward(inputs)

        # Compute loss
        loss = model.criterion(pred.view(-1), labels.view(-1))
        total_loss += loss.item()

    return total_loss / len(data)


def infer(model, data, threshold: float):
    # Évaluation du modèle
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in data:
            # Compute the prediction
            outputs = model(inputs)

            # Compute prediction
            pred = (torch.sigmoid(outputs).view(-1) > threshold).float()

            # Add prediction
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calcul du F1-score
    return f1_score(y_true, y_pred, average="macro")


def fit(model, train_loader, val_loader, epochs, threshold):
    # Loop on all epochs
    for epoch in range(epochs):
        model.train()

        # Mini-Batch
        for inputs, labels in train_loader:
            # Reset backward
            model.optimizer.zero_grad()

            # Model prediction
            pred = model.forward(inputs)

            # Compute loss
            loss = model.criterion(pred.view(-1), labels.view(-1))

            # Backpropagation
            loss.backward()

            # Change weights
            model.optimizer.step()

        train_loss = compute_loss(model, train_loader)
        val_loss = compute_loss(model, val_loader)

        f1_score_train = infer(model, train_loader, threshold)
        f1_score_val = infer(model, val_loader, threshold)

        print(f'\tEpoch [{epoch + 1}/{epochs}]')
        print(f'\tTrain - F1 score: {f1_score_train:.4f}, Loss: {train_loss:.4f}')
        print(f'\tValidation - F1 score: {f1_score_val:.4f}, Loss: {val_loss:.4f} \n')
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_f1": f1_score_train,
            "val_f1": f1_score_val,
        })


def k_cross_validation(x_inputs: torch.Tensor, y_labels: torch.Tensor, k=2, epochs=10,
                       batch_size=128) -> None:
    wandb.init(project="MLP")

    # Split data
    k_fold = KFold(n_splits=k, shuffle=True)
    threshold = 0.5

    # Compute weight for minority class
    class_0_count = (y_labels == 0).sum().item()
    class_1_count = y_labels.sum().item()
    minority_weight = torch.tensor([class_0_count / class_1_count], dtype=torch.float32)

    print(f"Minority weight: {minority_weight.item()}")

    # Iterate k-folds
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(x_inputs)):
        wandb.config.update({
            "learning_rate": 0.0001,
            "epochs": epochs,
            "k_folds": k
        })
        print(f"K-fold: [{fold + 1}/{k}]")

        #  Training set
        x_train = x_inputs[train_idx]
        y_train = y_labels[train_idx]

        # Validation set
        x_val = x_inputs[val_idx]
        y_val = y_labels[val_idx]

        # Set model
        model = MLPClassifier(x_train.shape[1])

        # Loss function
        model.criterion = nn.BCEWithLogitsLoss(weight=minority_weight)

        # Optimizer
        model.optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=0.01)
        # model.optimizer = optim.RMSprop(model.parameters())

        # Création du DataLoader
        train_dataset = TensorDataset(x_train, y_train)

        # Balance data
        sampler = create_balanced_sampler(y_train)

        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        # Création du DataLoader
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        fit(model, train_loader, val_loader, epochs, threshold)

        f1_score_train = infer(model, train_loader, threshold)
        print(f'Train - Fold [{fold + 1}/{k}], F1 score: {f1_score_train:.4f}')

        f1_score_val = infer(model, val_loader, threshold)
        print(f'Validation - Fold [{fold + 1}/{k}], F1 score: {f1_score_val:.4f}')

        threshold = plot_precision_recall_curve(model, val_loader)
        print(f'F1 Score - Fold [{fold + 1}/{k}], Best threshold: {threshold:.4f}')

        print("\n")

        wandb.log({
            "fold": fold + 1
        })

    wandb.finish()


def main():
    set_seed(1)

    # Load data
    inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
    labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]

    # Set tensor
    x_tensor = torch.tensor(inputs_documents, dtype=torch.float32)
    y_tensor = torch.tensor(labels_documents, dtype=torch.float32)

    k_cross_validation(x_tensor, y_labels=y_tensor)


if __name__ == '__main__':
    main()

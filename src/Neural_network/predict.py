import importlib
import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import src.config as config
from src.Neural_network.training import evaluation


def load_hyperparams(filename, file_path):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
    if base_path not in sys.path:
        sys.path.append(base_path)

    # Charger le module spécifié
    return importlib.import_module(filename)


def predict(output, hp_filename, hp_path):
    hyperparameters = load_hyperparams(hp_filename, hp_path)
    model, X_test, mean_f1_score = evaluation(hyperparameters, save_file=True)

    print(f"F1 score moyen obtenu : {mean_f1_score:.4f}")

    save_model_predictions(model, X_test, output, threshold=hyperparameters.infer_threshold,
                           batch_size=hyperparameters.batch_size)


def save_model_predictions(model, data_test, output_path, threshold, batch_size):
    # Initialisation
    model.eval()
    predictions = []

    # Define dataloader
    test_tensor = torch.tensor(data_test, dtype=torch.float32)
    test_dataset = TensorDataset(test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute prediction
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(config.DEVICE)
            outputs = model(inputs)
            pred = (torch.sigmoid(outputs).view(-1) > threshold).float()
            predictions.extend(pred.cpu().numpy())

    # Save model
    df_pred = pd.DataFrame(predictions, columns=['label'])
    df_pred['label'] = df_pred['label'].astype(int)
    df_pred.index.name = 'ID'
    df_pred.to_csv(f'{output_path}.csv')

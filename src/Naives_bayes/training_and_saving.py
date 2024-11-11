import importlib
import os
import sys

import pandas as pd
from src.Naives_bayes.utils import *


def load_hyperparams(filename, file_path):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
    if base_path not in sys.path:
        sys.path.append(base_path)

    # Charger le module spécifié
    return importlib.import_module(filename)


def train_model(hp, epochs=10):
    res = np.zeros((epochs, 3))
    for i in range(epochs):
        np.random.seed(i)
        print(f"Epoch [{i + 1} / {epochs}]:")
        mean_k_fold_accuracy, best_smooth, test_accuracy = k_fold_cross_validation(hp)
        res[i] = mean_k_fold_accuracy, best_smooth, test_accuracy

    best_test_index = np.argmax(np.bincount(res[:, 2].astype(int)))
    return res[best_test_index]


# Best: 0.4252481472265172
def save_prediction(output, hp_filename, hp_path):
    hyperparameters = load_hyperparams(hp_filename, hp_path)

    X_train, X_test = remove_low_high_frequency(hyperparameters.low_threshold, hyperparameters.high_threshold)

    # Train the model
    naives_bayes = NaiveBayesClassifier()

    naives_bayes.fit(X_train, config.LABELS_DOCUMENTS, hyperparameters.smoothing)

    pred = naives_bayes.predict(X_test)

    # Save the predictions
    df_pred = pd.DataFrame(pred, columns=['label'])
    df_pred.index.name = 'ID'
    df_pred.to_csv(f'{output}.csv')

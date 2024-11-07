import optuna
import numpy as np
import pandas as pd
from src.Naives_bayes.training_and_saving import train_model
from src.Naives_bayes.preprocessing import remove_stopwords, remove_low_frequency_v1, remove_low_frequency_v2

# Charger les données
inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
test_documents = np.load('../../data/data_test.npy', allow_pickle=True)
vocab = np.load('../../data/vocab_map.npy', allow_pickle=True)

# Définir la fonction objectif pour Optuna
def objective(trial):
    # Hyperparamètres
    smoothing = trial.suggest_float('smoothing', 0.0, 1.0)
    threshold = trial.suggest_int('threshold', 1, 20)

    # Prétraitement
    processed_docs = inputs_documents
    processed_docs, _ = remove_low_frequency_v2(processed_docs, test_documents, labels_documents, threshold=threshold)
    # Entraînement et évaluation du modèle
    mean_f1_score = train_model(processed_docs, labels_documents, smoothing=smoothing, epochs=50)[2]

    # Retourner le score pour l'optimisation
    return mean_f1_score


# Lancer l'optimisation bayésienne
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

# Meilleurs hyperparamètres trouvés
print("Best hyperparameters: ", study.best_params)
print("Best F1 score: ", study.best_value)

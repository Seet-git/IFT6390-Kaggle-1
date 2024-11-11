import pandas as pd
import numpy as np
import src.config as config
from src.Neural_network.bayesian_optimization import bayesian_optimization
from src.Neural_network.predict import predict

# <----- CHANGE THIS ------>

# Activate WANDB
config.WANDB_ACTIVATE = False  # Note: login before activating

# Data path
data_path = "../../data/"

# BAYESIAN OPTIMISATION
# -------------------------------------------------------------
n_trials = 2

# Login mysql
config.USER = "optuna_seet"
config.PASSWORD = "@g3NYkke*eAFRs"
config.DATABASE_NAME = "optuna_MLP"
config.ENDPOINT = "localhost"  # Local
# ENDPOINT = 1.tcp.eu.ngrok.io:3791 # ngrok (LOTR)

config.OUTPUT_HP_FILENAME = "hp_mlp_template"
config.OUTPUT_HP_PATH = "../../hyperparameters/"

# PREDICTIONS
# -------------------------------------------------------------
# Hyperparameters
hp_filename = "hp_mlp_template"
hp_path = "../../hyperparameters/"

# Output prediction
prediction_filename = "MLP_optimized"
prediction_path = "../../output/"


def main():
    print(f"device: {config.DEVICE}")

    # Load data
    config.INPUTS_DOCUMENTS = np.load(f'{data_path}data_train.npy', allow_pickle=True)
    config.LABELS_DOCUMENTS = pd.read_csv(f'{data_path}label_train.csv').to_numpy()[:, 1]
    config.TEST_DOCUMENTS = np.load(f'{data_path}data_test.npy', allow_pickle=True)

    # Bayesian optimization
    bayesian_optimization(n_trials=n_trials)

    output = f"{prediction_path}{prediction_filename}"

    # Predict
    predict(output=output, hp_filename=hp_filename, hp_path=hp_path)


if __name__ == '__main__':
    main()
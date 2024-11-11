from src.Naives_bayes.training_and_saving import save_prediction
from src.Naives_bayes.optimizer import optimize
import src.config as config
import numpy as np
import pandas as pd

config.ALGORITHM = "Naives_bayes"

# <----- CHANGE THIS ------>

# Data path
data_path = "../../data/"

# BAYESIAN OPTIMISATION
# -------------------------------------------------------------
n_trials = 2

config.OUTPUT_HP_FILENAME = f"hp_{config.ALGORITHM}"
config.OUTPUT_HP_PATH = "../../hyperparameters/"
config.LOG_PATH = "./log"

# PREDICTIONS
# -------------------------------------------------------------
# Hyperparameters
hp_filename = f"hp_{config.ALGORITHM}"
hp_path = "../../hyperparameters/"

# Output prediction
prediction_filename = f"{config.ALGORITHM}_pred"
prediction_path = "../../output/"


def NaiveBayesClassifier():
    print(f"device: {config.DEVICE}")

    # Load data
    config.INPUTS_DOCUMENTS = np.load(f'{data_path}data_train.npy', allow_pickle=True)
    config.LABELS_DOCUMENTS = pd.read_csv(f'{data_path}label_train.csv').to_numpy()[:, 1]
    config.TEST_DOCUMENTS = np.load(f'{data_path}data_test.npy', allow_pickle=True)

    optimize(n_trials=n_trials)

    output = f"{prediction_path}{prediction_filename}"

    # Predict
    save_prediction(output=output, hp_filename=hp_filename, hp_path=hp_path)

if __name__ == "__main__":
    NaiveBayesClassifier()
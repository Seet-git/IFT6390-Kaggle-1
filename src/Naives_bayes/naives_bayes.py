import os

from src.Naives_bayes.predict import save_prediction
from src.Naives_bayes.bayesian_optimization import optimize
import config
import numpy as np
import pandas as pd


def main():
    if config.ALGORITHM != "Naives_bayes":
        raise ValueError("Bad ALGORITHM value")

    print(f"device: {config.DEVICE}")
    config.LOG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs/")

    # Load data
    config.INPUTS_DOCUMENTS = np.load(f"../../{config.DATA_PATH}data_train.npy", allow_pickle=True)
    config.LABELS_DOCUMENTS = pd.read_csv(f"../../{config.DATA_PATH}label_train.csv").to_numpy()[:, 1]
    config.TEST_DOCUMENTS = np.load(f"../../{config.DATA_PATH}data_test.npy", allow_pickle=True)
    config.VOCAB = np.load(f"../../{config.DATA_PATH}vocab_map.npy", allow_pickle=True)

    optimize(n_trials=config.N_TRIALS)

    # Predict
    save_prediction()


if __name__ == '__main__':
    main()

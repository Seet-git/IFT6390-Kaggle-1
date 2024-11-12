import pandas as pd

import config
from src.Naives_bayes.utils import *
from src.scripts.load import load_hyperparams
from src.scripts.word_cloud import generate_word_cloud


def train_model(hp, epochs=10):
    res = np.zeros((epochs, 3))
    for i in range(epochs):
        np.random.seed(i)
        print(f"Epoch [{i + 1} / {epochs}]:")
        mean_k_fold_accuracy, best_smooth, test_accuracy = k_fold_cross_validation(hp)
        res[i] = mean_k_fold_accuracy, best_smooth, test_accuracy

    best_test_index = np.argmax(np.bincount(res[:, 2].astype(int)))
    return res[best_test_index]


def save_prediction():
    hyperparameters = load_hyperparams()

    X_train, X_test = remove_low_high_frequency(hyperparameters.low_threshold, hyperparameters.high_threshold)

    generate_word_cloud()

    # Train the model
    naives_bayes = NaiveBayesClassifier()

    naives_bayes.fit(X_train, config.LABELS_DOCUMENTS, hyperparameters.smoothing)

    pred = naives_bayes.predict(X_test)

    # Save the predictions
    df_pred = pd.DataFrame(pred, columns=['label'])
    df_pred.index.name = 'ID'
    df_pred.to_csv(f'../../{config.PREDICTION_PATH}/{config.ALGORITHM}/{config.PREDICTION_FILENAME}.csv')

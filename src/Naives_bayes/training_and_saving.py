import pandas as pd
from src.Naives_bayes.utils import *


def train_model(train_inputs, labels_inputs, smoothing, epochs=10):
    res = np.zeros((epochs, 3))
    for i in range(epochs):
        np.random.seed(i)
        print(f"Epoch [{i + 1} / {epochs}]:")
        mean_k_fold_accuracy, best_smooth, test_accuracy = k_fold_cross_validation(train_inputs,
                                                                                   labels_inputs, smoothing,
                                                                                   k=0)
        res[i] = mean_k_fold_accuracy, best_smooth, test_accuracy

    best_test_index = np.argmax(np.bincount(res[:, 2].astype(int)))
    return res[best_test_index]


# Best: 0.4252481472265172
def save_prediction(train_inputs, train_labels, data_test, hp):
    name = str(input("File name (Press enter > default): "))
    if name == '': name = "naive_bayes_best"

    # Train the model
    naives_bayes = NaiveBayesClassifier()

    naives_bayes.fit(train_inputs, train_labels, hp)

    pred = naives_bayes.predict(data_test)

    # Save the predictions
    df_pred = pd.DataFrame(pred, columns=['label'])
    df_pred.index.name = 'ID'
    df_pred.to_csv(f'../../output/{name}.csv')

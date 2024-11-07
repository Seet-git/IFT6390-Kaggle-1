import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from nltk.corpus import stopwords
from src.Naives_bayes.preprocessing import *

sw_nltk = stopwords.words('english')


class NaiveBayesClassifier:

    def __init__(self):
        self.p_x_given_y = None
        self.p_y = None
        self.y_train = None
        self.x_train = None
        self.y_labels = None

    def fit(self, x, y, smoothing: float = 1):
        self.y_labels = np.unique(y)
        self.x_train = x
        self.y_train = y

        # Calculer P(Y) = nb Yi / nb Y

        # p_y: List(n_class)
        n_class = len(self.y_labels)
        self.p_y = np.zeros(n_class)

        for classes in range(n_class):
            current_class = self.y_labels[classes]
            nb_docs = self.x_train.shape[0]
            self.p_y[classes] = np.sum(self.y_train == current_class) / nb_docs

        # Calculer P(Xi|Y) = nb Xi | Y / nb X | Y

        # p_x_given_y: Matrix(n_class, n_features)
        n_features = self.x_train.shape[1]
        self.p_x_given_y = np.zeros((n_class, n_features))

        for classes in range(n_class):
            # Calculate \sum of all Xi given a class y
            current_class = self.y_labels[classes]
            self.p_x_given_y[classes] = np.sum(self.x_train[self.y_train == current_class], axis=0) + smoothing

            # Calculate \sum of all X for a given class y
            nb_inputs_class = np.sum(self.p_x_given_y[classes]) + n_features * smoothing

            # Calculate P(Xi) given a class y
            self.p_x_given_y[classes] = self.p_x_given_y[classes] / nb_inputs_class

    def predict(self, test_input):
        # Apply log to avoid underflow
        log_p_y = np.log(self.p_y)
        log_p_x_given_y = np.log(self.p_x_given_y)

        res = []

        # Calculate Log P(Y|X) = Log P(Y) * \prod P(X|Y)^x = P(Y) + sum P(X|Y) * x
        for point in test_input:
            res_class = np.dot(point, log_p_x_given_y.T) + log_p_y
            pred_class = np.argmax(res_class)
            res.append(self.y_labels[pred_class])
        return res


def k_fold_split(train_set: tuple, index_tab, fold_size: int, k_index: int):
    """
    Split the training set into k folds
    :param index_tab:
    :param fold_size:
    :param train_set: Tuple of inputs and labels
    :param k_index:
    :return: List of k folds
    """
    set_inputs = train_set[0]
    set_labels = train_set[1]

    # Start and end index of the fold
    start = k_index * fold_size
    end = (k_index + 1) * fold_size

    # Validation set
    valid_index = index_tab[start:end]
    x_valid = set_inputs[valid_index, :]
    y_valid = set_labels[valid_index]

    # Training set
    test_index = np.concatenate([index_tab[:start], index_tab[end:]])
    x_train = set_inputs[test_index, :]
    y_train = set_labels[test_index]

    return x_train, y_train, x_valid, y_valid


def split_dataset(inputs_train: np.array, labels_train: np.array):
    """
    Split the dataset into training and test set
    :param inputs_train:
    :param labels_train:
    :return:
    """
    # Shuffle the dataset
    indices = np.arange(len(inputs_train))
    np.random.shuffle(indices)

    inputs_shuffled = inputs_train[indices]
    labels_shuffled = labels_train[indices]

    # Get the size of the training set
    train_size = int(np.ceil(0.8 * len(inputs_train)))

    # Train set
    set_train_inputs = inputs_shuffled[0:train_size, :]
    set_train_labels = labels_shuffled[0:train_size]

    # Test set
    set_test_inputs = inputs_shuffled[train_size:, :]
    set_test_labels = labels_shuffled[train_size:]

    return (set_train_inputs, set_train_labels), (set_test_inputs, set_test_labels)


def compute_f1_score_macro(y_true, y_pred):
    """
    Calcule le F1 score macro.
    :param y_true: Labels réels
    :param y_pred: Prédictions du modèle
    :return: F1 score macro
    """
    # Récupérer les labels uniques
    labels = np.unique(y_true)

    # Initialiser les compteurs
    f1_scores = []

    for label in labels:
        # Vrais positifs
        vrai_pos = np.sum((y_true == label) & (y_pred == label))

        # Faux positifs
        faux_pos = np.sum((y_true != label) & (y_pred == label))

        # Faux négatifs
        faux_neg = np.sum((y_true == label) & (y_pred != label))

        # Accuracy et recall
        accuracy = vrai_pos / (vrai_pos + faux_pos) if (vrai_pos + faux_pos) > 0 else 0
        recall = vrai_pos / (vrai_pos + faux_neg) if (vrai_pos + faux_neg) > 0 else 0

        # F1 score
        f1 = (2 * accuracy * recall) / (accuracy + recall) if (accuracy + recall) > 0 else 0
        f1_scores.append(f1)

    # F1 score macro
    f1_macro = np.mean(f1_scores)
    return f1_macro


# **Fonction Objectif pour Hyperopt**
def objective(params):
    inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
    labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
    vocab = np.load('../../data/vocab_map.npy', allow_pickle=True)
    smoothing = params['smoothing']
    inputs_documents = remove_stopwords(vocab, inputs_documents)
    inputs_documents = remove_low_frequency_v2(inputs_documents, labels_documents, 10)

    # Splitter le dataset pour créer une validation set
    train_set, test_set = split_dataset(inputs_documents, labels_documents)

    x_train, y_train = train_set
    x_valid, y_valid = test_set

    # Créer et entraîner le modèle Naive Bayes
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(x_train, y_train, smoothing=smoothing)

    # Prédire sur le jeu de validation
    y_pred = nb_classifier.predict(x_valid)

    # Calculer le F1 Score macro
    score = compute_f1_score_macro(y_valid, y_pred)

    # Hyperopt minimise la fonction objectif, donc on retourne -score (car F1 doit être maximisé)
    return {'loss': -score, 'status': STATUS_OK}


def main():
    space = {
        'smoothing': hp.uniform('smoothing', 0.2, 0.5),
        'inputs': np.load('../../data/data_train.npy', allow_pickle=True),
        'labels': pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
    }

    # Trials permet de suivre les essais effectués
    trials = Trials()

    # Optimisation avec Hyperopt
    best = fmin(fn=objective,  # fonction objectif
                space=space,  # espace de recherche
                algo=tpe.suggest,  # algorithme d'optimisation bayésienne
                max_evals=10,  # nombre maximum d'évaluations
                trials=trials)  # pour suivre les résultats

    print(f"Meilleur smoothing trouvé: {best['smoothing']}")


if __name__ == '__main__':
    main()

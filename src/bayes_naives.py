# Import nltk and download stopwords
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from numpy.core.fromnumeric import argmax

nltk.download('stopwords')
sw_nltk = stopwords.words('english')
steemer = PorterStemmer()


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
            nb_inputs_class = np.sum(self.p_x_given_y[classes]) + n_features

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
            pred_class = argmax(res_class)
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


def k_fold_cross_validation(inputs_train, labels_train, k=5):
    """
    Validation croisée en k-fold
    :param inputs_train:
    :param labels_train:
    :param k:
    :return:
    """
    # Split the dataset into train and test set
    train_set, test_set = split_dataset(inputs_train, labels_train)

    n = len(train_set[0])  # Number of samples
    fold_size = n // k  # Size of each fold

    # Shuffle the dataset
    index_tab = np.arange(n)
    np.random.shuffle(index_tab)

    smooth_tab = [0, 0.01, 0.02, 0.05, 0.07, 0.1, 1]

    # List of accuracy for each fold
    accuracy_k_fold = np.zeros((len(smooth_tab), k))
    smooth_scores = np.zeros(k)
    for i in range(k):
        print(f"Fold {i + 1} / {k}")

        # Split the training set into k folds
        x_train, y_train, x_valid, y_valid = k_fold_split(train_set, index_tab, fold_size, i)

        best_accuracy = 0
        for smooth in smooth_tab:

            # Train the model
            nb_classifier = NaiveBayesClassifier()
            nb_classifier.fit(x_train, y_train, smoothing=smooth)
            y_pred = nb_classifier.predict(x_valid)

            # Compute the accuracy
            accuracy = compute_f1_score_macro(y_valid, y_pred)

            # Append the accuracy
            accuracy_k_fold[smooth_tab.index(smooth), i] = accuracy

            # Save the best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                smooth_scores[i] = smooth

            print(f"Accuracy for fold {i + 1} and smoothing {smooth}: {accuracy}")

    # Find the most frequent smoothing
    best_smooth_index = np.argmax(np.bincount(smooth_scores.astype(int)))
    best_smooth = smooth_scores[best_smooth_index]

    # Moyenne de l'accuracy sur tous les folds
    mean_k_fold_accuracy = np.mean(accuracy_k_fold[best_smooth_index])

    print(f"Mean Accuracy across all folds: {mean_k_fold_accuracy} with smoothing {best_smooth}")

    # Train the model on the whole training sets
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(train_set[0], train_set[1])

    # Test the model
    y_pred = nb_classifier.predict(test_set[0])
    test_accuracy = compute_f1_score_macro(test_set[1], y_pred)
    print(f"Accuracy on test set: {test_accuracy}")

    return mean_k_fold_accuracy, best_smooth, test_accuracy


def remove_stopwords(vocab: np.array, inputs_documents: np.array):
    """
    Filter the document to remove stopwords
    :param vocab: vocabulary of the documents
    :param inputs_documents: Inputs documents
    :return: Vocabulary without stopwords
    """

    # Remove stopwords
    clean_index = [word for word in range(len(vocab)) if vocab[word].lower() in sw_nltk]
    # stemmer.stem
    # Filter the inputs
    return np.delete(inputs_documents, clean_index, axis=1)


def tf_idf(inputs_documents: np.array):
    tf = (inputs_documents + 0.1) / np.sum(inputs_documents, axis=1, keepdims=True)

    # Calcul de l'IDF : ajoute 1 pour éviter log(0)
    idf = np.log((inputs_documents.shape[0] + 1) / (1 + np.sum(inputs_documents > 0, axis=0))) + 1

    return tf * idf


def main():
    inputs_documents = np.load('../data/data_train.npy', allow_pickle=True)
    labels_documents = pd.read_csv('../data/label_train.csv').to_numpy()[:, 1]
    test_documents = np.load('../data/data_test.npy', allow_pickle=True)
    vocab = np.load('../data/vocab_map.npy', allow_pickle=True)

    # Remove stopwords
    inputs_documents = remove_stopwords(vocab, inputs_documents)
    test_documents = remove_stopwords(vocab, test_documents)

    # # TF-IDF
    # inputs_documents = tf_idf(inputs_documents)
    # test_documents = tf_idf(test_documents)

    # Cross validation
    iterate = 1
    res = np.zeros((iterate, 3))
    for i in range(iterate):
        np.random.seed(i)
        print(f"Iteration {i + 1}: \n")
        mean_k_fold_accuracy, best_smooth, test_accuracy = k_fold_cross_validation(inputs_documents, labels_documents,
                                                                                   k=5)
        res[i] = mean_k_fold_accuracy, best_smooth, test_accuracy

    best_test_index = np.argmax(np.bincount(res[:, 2].astype(int)))
    print("FINAL: ", res[best_test_index])

    # Train the model
    naives_bayes = NaiveBayesClassifier()
    naives_bayes.fit(inputs_documents, labels_documents, smoothing=0.5)
    pred = naives_bayes.predict(test_documents)

    np.savetxt('bayes_naives.csv', pred, fmt='%d', delimiter=',')


if __name__ == '__main__':
    main()

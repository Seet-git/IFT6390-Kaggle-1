from src.Naives_bayes.Naives_Bayes import *
import numpy as np


def k_fold_split(inputs_train, labels_train, index_tab, fold_size: int, k_index: int):
    """
    Split the training set into k folds
    :param labels_train:
    :param inputs_train:
    :param index_tab:
    :param fold_size:
    :param k_index:
    :return: List of k folds
    """

    # Start and end index of the fold
    start = k_index * fold_size
    end = (k_index + 1) * fold_size

    # Validation set
    valid_index = index_tab[start:end]
    x_valid = inputs_train[valid_index, :]
    y_valid = labels_train[valid_index]

    # Training set
    test_index = np.concatenate([index_tab[:start], index_tab[end:]])
    x_train = inputs_train[test_index, :]
    y_train = labels_train[test_index]

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
    # train_size = int(np.ceil(0.8 * len(inputs_train)))
    train_size = len(inputs_train) - 2355

    # Train set
    set_train_inputs = inputs_shuffled[0:train_size, :]
    set_train_labels = labels_shuffled[0:train_size]

    # Test set
    set_test_inputs = inputs_shuffled[train_size:, :]
    set_test_labels = labels_shuffled[train_size:]

    return set_train_inputs, set_train_labels, set_test_inputs, set_test_labels


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


def k_fold_cross_validation(inputs_documents, labels_documents, smoothing, k=5):
    """
    Validation croisée en k-fold
    :param inputs_documents:
    :param labels_documents:
    :param k:
    :return:
    """
    # Split the dataset into train and test set
    inputs_train, labels_train, inputs_test, labels_test = split_dataset(inputs_documents, labels_documents)
    n = len(inputs_train)  # Number of samples

    # Shuffle the dataset
    index_tab = np.arange(n)
    np.random.shuffle(index_tab)

    smooth_tab = [smoothing]

    fold_size = 0
    accuracy_k_fold = np.zeros(1)
    smooth_scores = np.zeros(1)
    if k > 1:
        fold_size = n // k  # Size of each fold

        # List of F1_score for each fold
        accuracy_k_fold = np.zeros((len(smooth_tab), k))
        smooth_scores = np.zeros(k)

    for i in range(k):
        print(f"\tFold {i + 1} / {k}")

        # Split the training set into k folds
        x_train, y_train, x_valid, y_valid = k_fold_split(inputs_train, labels_train, index_tab, fold_size, i)

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

            print(f"\tF1 Score {accuracy} - Fold [{i + 1}/{k}], Smoothing {smooth} \n")

    # Find the most frequent smoothing
    best_smooth_index = np.argmax(np.bincount(smooth_scores.astype(int)))
    best_smooth = smooth_scores[best_smooth_index]

    # Moyenne de l'accuracy sur tous les folds
    mean_k_fold_accuracy = np.mean(accuracy_k_fold[best_smooth_index])

    #print(f"Mean F1-score across all folds: {mean_k_fold_accuracy}, Best smoothing {best_smooth}")

    # Train the model on the whole training sets
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.fit(inputs_train, labels_train, smoothing=smooth_tab[0])

    # Test test set
    y_pred = nb_classifier.predict(inputs_test)
    test_accuracy = compute_f1_score_macro(labels_test, y_pred)
    print(f"F1 Score - Test set: {test_accuracy}\n")
    return mean_k_fold_accuracy, best_smooth, test_accuracy

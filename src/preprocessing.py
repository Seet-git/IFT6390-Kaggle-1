import numpy as np
from nltk.corpus import stopwords

import src.config as config

sw_nltk = stopwords.words('english')


def remove_stopwords(vocab: np.array):
    """
    Filter the document to remove stopwords
    :param vocab: vocabulary of the documents
    :return: Vocabulary without stopwords
    """

    # Remove stopwords
    clean_index = [word for word in range(len(vocab)) if vocab[word].lower() in sw_nltk]

    # Filter the inputs
    return np.delete(config.INPUTS_DOCUMENTS, clean_index, axis=1)


def tf_idf(docs):
    # Diviser chaque valeur par le total des mots dans le doc
    tf = docs / np.sum(docs, axis=1, keepdims=True)

    # Calcul de l'IDF
    N = docs.shape[0]  # Nombre total de documents
    df = np.sum(docs > 0, axis=0)  # Nombre de documents contenant chaque mot
    idf = np.log(N / (df + 1))  # On ajoute 1 pour éviter la division par 0

    # Calcul du TF-IDF
    tf_idf = tf * idf
    return tf_idf


def remove_low_high_frequency(low_threshold, high_threshold):
    # Count each word
    occurrence_0 = np.sum(config.INPUTS_DOCUMENTS[config.LABELS_DOCUMENTS == 0], axis=0)
    occurrence_1 = np.sum(config.INPUTS_DOCUMENTS[config.LABELS_DOCUMENTS == 1], axis=0)

    # Différence des occurrences entre chaque classe
    difference = np.abs(occurrence_0 - occurrence_1)

    # Supprime les mots en dessous du seuil d'apparition
    delete_words_low = [x for x in range(len(difference)) if difference[x] < low_threshold]

    X_train = np.delete(config.INPUTS_DOCUMENTS, delete_words_low, axis=1)
    X_test = np.delete(config.TEST_DOCUMENTS, delete_words_low, axis=1)

    if high_threshold == 0:
        return X_train, X_test

    occurrence_0 = np.sum(X_train[config.LABELS_DOCUMENTS == 0], axis=0)
    occurrence_1 = np.sum(X_train[config.LABELS_DOCUMENTS == 1], axis=0)

    # Différence des occurrences entre chaque classe
    difference = np.abs(occurrence_0 - occurrence_1)

    sorted_index = np.argsort(difference)

    delete_words_high = sorted_index[-high_threshold:]

    # Supprime les mots en dessous du seuil d'apparition
    return np.delete(X_train, delete_words_high, axis=1), np.delete(X_test, delete_words_high, axis=1)

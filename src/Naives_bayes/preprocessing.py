import numpy as np
from nltk.corpus import stopwords

sw_nltk = stopwords.words('english')


def remove_stopwords(vocab: np.array, inputs_documents: np.array):
    """
    Filter the document to remove stopwords
    :param vocab: vocabulary of the documents
    :param inputs_documents: Inputs documents
    :return: Vocabulary without stopwords
    """

    # Remove stopwords
    clean_index = [word for word in range(len(vocab)) if vocab[word].lower() in sw_nltk]

    # Filter the inputs
    return np.delete(inputs_documents, clean_index, axis=1)


def tf_idf(docs):
    # Calcul du TF pour chaque document
    tf = docs / np.sum(docs, axis=1, keepdims=True)  # Diviser chaque valeur par le total des mots dans le doc

    # Calcul du IDF (Inverse Document Frequency)
    N = docs.shape[0]  # Nombre total de documents
    df = np.sum(docs > 0, axis=0)  # Nombre de documents contenant chaque mot
    idf = np.log(N / (df + 1))  # On ajoute 1 pour éviter la division par 0

    # Calcul du TF-IDF
    tf_idf = tf * idf
    return tf_idf

def remove_low_frequency_v2(inputs_documents: np.array, test_documents: np.array, labels_documents: np.array, threshold=1):

    # Compte each words
    occurrence_0 = np.sum(inputs_documents[labels_documents == 0], axis=0)
    occurrence_1 = np.sum(inputs_documents[labels_documents == 1], axis=0)

    # Différence des occurrences entre chaque classe
    difference = np.abs(occurrence_0 - occurrence_1)

    # Supprime les mots en dessous du seuil d'apparition
    delete_words = [x for x in range(len(difference)) if difference[x] < threshold]
    return np.delete(inputs_documents, delete_words, axis=1), np.delete(test_documents, delete_words, axis=1)


def remove_low_frequency_v1(inputs_documents: np.array, threshold=1):

    # Compte each words
    occurrence = np.sum(inputs_documents, axis=0)
    delete_words = [x for x in range(len(occurrence)) if occurrence[x] < threshold]
    return np.delete(inputs_documents, delete_words, axis=1)



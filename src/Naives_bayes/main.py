from torch.nn.functional import threshold

from src.Naives_bayes.training_and_saving import *
from src.Naives_bayes.preprocessing import *
import numpy as np
import pandas as pd


def main():
    inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
    labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
    test_documents = np.load('../../data/data_test.npy', allow_pickle=True)

    # Hyper-parameters
    threshold = 5
    smoothing = 0.005400920956062999
    inputs_documents, test_documents = remove_low_frequency_v2(inputs_documents, test_documents, labels_documents, threshold=threshold)


    # Train model
    res = train_model(inputs_documents, labels_documents, smoothing)
    print("FINAL RESULTS: ", res)

    # Predict
    #save_prediction(inputs_documents, labels_documents, test_documents, smoothing)


if __name__ == '__main__':
    main()

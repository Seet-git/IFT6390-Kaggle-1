from src import *


def main():
    inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
    labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
    test_documents = np.load('../../data/data_test.npy', allow_pickle=True)
    # vocab = np.load('../../../data/vocab_map.npy', allow_pickle=True)

    hp = {'smoothing': 0.4252481472265172}
    save_prediction(inputs_documents, labels_documents, test_documents, hp['smoothing'])


if __name__ == '__main__':
    main()

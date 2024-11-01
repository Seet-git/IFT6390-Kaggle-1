import numpy as np
import pandas as pd

import wandb


class SVM:
    def __init__(self, vocab):
        self.__vocab = vocab
        self.y_train = None
        self.x_train = None
        self.label_list = None

    def soft_margin(self, x_inputs, y_labels, w, C):
        """
        Minimise g(x) = 1/2 * ||w||² + C x \sum_i=1^n (Hinge loss)
        :param x_inputs:
        :param y_labels:
        :param w:
        :param C:
        :return:
        """

        n = x_inputs.shape[0]
        loss = 0

        # For all point n
        for i in range(n):
            # Loss calculate
            f_x = np.dot(w, x_inputs[i, :])  # Calcul f(x)
            loss += np.maximum(0, 1 - f_x * y_labels[i])  # Logistic Hinge loss between f(x) and y
            # print("loss: ", np.maximum(0, 1 - f_x * y_labels[i]))

        # Add the regularization
        omega = 1 / 2 * np.power(np.linalg.norm(w), 2)

        return 1 / n * (omega + C * loss)

    def compute_gradient(self, x_batch, y_batch, w, C):
        """
        Adjust weight: w - C * \sum_i=1^n y_i * x_i -> si product 1- y_i f(x)
        :param x_batch:
        :param y_batch:
        :param w:
        :param C:
        :return:
        """
        num_features = x_batch.shape[1]
        grad = np.zeros(num_features)
        n = x_batch.shape[0]
        k = 0
        # Calculate the gradient
        for i in range(n):

            # Calculate f(x)
            f_xi = np.dot(w, x_batch[i])
            # print("F", f_xi)

            if 1 - y_batch[i] * f_xi > 0:
                k += 1
                # Derivative hinge loss
                grad += y_batch[i] * x_batch[i]
        grad = 1 / n * (w - C * grad)
        return grad, k

    def fit(self, training_inputs: np.array, training_labels: np.array, epochs, learning_rate, C, batch_size):
        # Training 80% - Validation 20%
        train_size = int(np.ceil(0.8 * training_inputs.shape[0]))

        # Training data
        x_train = training_inputs[0:train_size, :]
        y_train = training_labels[0:train_size]

        # Validation data
        x_val = training_inputs[train_size:, :]
        y_val = training_labels[train_size:]

        num_features = x_train.shape[1]
        w = np.zeros(num_features)

        # Dictionaries to store the metrics
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        for epoch in range(epochs):
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)

            # Mini-batch training
            for i in range(0, x_train.shape[0], batch_size):
                batch_indices = indices[i:i + batch_size]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]

                # Compute gradients and update weights
                grad, k = self.compute_gradient(x_batch, y_batch, w, C)
                w -= learning_rate * grad
                # print("K: ", k/x_batch.shape[0])
            exit(self.soft_margin(x_train, y_train, w, C))

            # Compute training and validation loss and accuracy
            train_loss = self.soft_margin(x_train, y_train, w, C)
            val_loss = self.soft_margin(x_val, y_train, w, C)

            # TODO
            # train_acc = self.compute_accuracy(X_train, y_train, w)
            # val_acc = self.compute_accuracy(X_val, y_val, w)

            # Store the metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            # history['train_acc'].append(train_acc)
            # history['val_acc'].append(val_acc)

            # wandb.log({
            #     "train_loss": train_loss,
            #     "val_loss": val_loss,
            #     "epoch": epoch
            # })

            print(f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")

        return history['train_loss'][-1], history['val_loss'][-1]


def split_dataset(inputs_documents: np.array, labels_documents: np.array):
    # Training set
    train_size = int(np.ceil(0.8 * len(inputs_documents)))
    training_inputs = inputs_documents[0:train_size, :]
    training_labels = labels_documents[0:train_size]

    # Test set
    x_test = inputs_documents[train_size:, :]
    y_test = labels_documents[train_size:]

    return training_inputs, training_labels, x_test, y_test


def main():
    # Load data
    inputs_documents = np.load('../../data/data_train.npy', allow_pickle=True)
    labels_documents = pd.read_csv('../../data/label_train.csv').to_numpy()[:, 1]
    labels_documents = np.where(labels_documents == 0, -1, 1)

    vocab = np.load('../../data/vocab_map.npy', allow_pickle=True)

    # Split train and test
    training_inputs, training_labels, x_test, y_test = split_dataset(inputs_documents, labels_documents)

    # Initialize SVM
    svm_model = SVM(vocab)

    epochs_tab = np.array([10, 50, 100, 200], dtype=int)
    learning_rates_tab = np.array([0.0001, 0.001, 0.01, 0.1])
    C_tab = np.array([0.01, 0.1, 1, 10, 100])
    batch_sizes_tab = np.array([1, 8, 32, 64], dtype=int)

    # Utiliser meshgrid pour générer la grille de combinaisons
    grid = np.array(np.meshgrid(epochs_tab, learning_rates_tab, C_tab, batch_sizes_tab, indexing='ij'),
                    dtype=object).T.reshape(-1, 4)

    history_tab = []
    compteur = 1
    for i in grid:
        # wandb.init(project="SVM-hp-tuning-1", name=f"{compteur}", config={
        #     "epochs": i[0],
        #     "learning_rate": i[1],
        #     "C": i[2],
        #     "batch_size": i[3]
        # }, allow_val_change=True)

        # Fit the model
        print(f"epoch: {i[0]}, lr : {i[1]}, C: {i[2]}, batch_size: 28")
        train_loss, val_loss = svm_model.fit(training_inputs, training_labels, i[0], i[1], 1, i[3])

        # Log the results
        #wandb.finish()

        print(f"{compteur}/{grid.shape[0]}")
        compteur += 1
    print("Done")
    print(history_tab)


if __name__ == '__main__':
    main()

    # # TODO
    # def compute_accuracy(self, X, y, w):
    #     """
    #     Computes the accuracy of predictions using the given weight matrix.
    #
    #     Input:
    #     - X: Feature matrix of shape (n_samples, n_features).
    #     - y: True label vector of shape (n_samples,).
    #     - w: Weight matrix of shape (n_features, n_classes).
    #
    #     Process:
    #     - Predicts the labels for the given feature matrix using the weights.
    #     - Compares predicted labels with true labels to compute accuracy.
    #
    #     Output:
    #     - Returns the accuracy score (float) as a percentage of correct predictions.
    #     """
    #     raise NotImplementedError
    #
    #
    # # TODO
    # def infer(self, X, w):
    #     raise NotImplementedError

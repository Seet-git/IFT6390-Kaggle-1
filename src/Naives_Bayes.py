import numpy as np


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

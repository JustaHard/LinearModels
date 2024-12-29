import numpy as np

class LinearRegression:
    def __init__(self, learning_rate, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            diff = y_pred - y

            dw = np.dot(X.T, (diff)) / n_samples
            db = np.sum(diff) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def mse(self, y_true, y_pred):
        error = np.mean((y_true - y_pred)**2)
        return error

class LogisticRegression:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, learning_rate, threshold, iterations=100, L1_reg=0, L2_reg=0):
        n_samples, n_features = X.shape
        self.threshold = threshold
        self.n_labels = len(np.unique(y))
        self.weights, self.bias = np.zeros(n_features), 0

        for i in range(iterations):
            y_pred = self.sigmoid(X.dot(self.weights)+self.bias)
            diff = y - y_pred

            dw = X.T.dot(diff) / n_samples + L1_reg * np.sign(self.weights) + 2 * L2_reg * self.weights
            db = np.sum(diff) / n_samples

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        return np.where(self.sigmoid(X.dot(self.weights)+self.bias)<self.threshold, 1, 0)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def confusion_matrix(self, y_true, y_pred):
        matrix = np.zeros((self.n_labels, self.n_labels), dtype=int)

        for true, pred in zip(y_true, y_pred):
            matrix[true, pred] += 1

        return matrix

    def find_TP(self, y_true, y_pred, positive_class):
        return np.sum((y_pred==positive_class) & (y_true==positive_class))

    def find_FP(self, y_true, y_pred, positive_class):
        return np.sum((y_pred==positive_class) & (y_true!=positive_class))

    def find_FN(self, y_true, y_pred, positive_class):
        return np.sum((y_pred!=positive_class) & (y_true==positive_class))

    def precision(self, y_true, y_pred, positive_class=1):
        TP = self.find_TP(y_true, y_pred, positive_class)
        FP = self.find_FP(y_true, y_pred, positive_class)

        if TP + FP == 0:
            return 0.0
        return TP / (TP + FP)

    def recall(self, y_true, y_pred, positive_class=1):
        TP = self.find_TP(y_true, y_pred, positive_class)
        FN = self.find_FN(y_true, y_pred, positive_class)

        if TP + FN == 0:
            return 0.0
        return TP / (TP + FN)

    def f1(self, y_true, y_pred, positive_class=1):
        recall = self.recall(y_true, y_pred, positive_class)
        precision = self.precision(y_true, y_pred, positive_class)

        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    def f_beta(self, y_true, y_pred, beta, positive_class=1):
        recall = self.recall(y_true, y_pred, positive_class)
        precision = self.precision(y_true, y_pred, positive_class)

        if recall + precision == 0:
            return 0.0
        return (beta ** 2 + 1) * recall * precision/(recall + beta ** 2 * precision)
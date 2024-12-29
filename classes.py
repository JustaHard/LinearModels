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
    def __init__(self, learning_rate, threshold, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.threshold = threshold

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0

        for i in range(self.iterations):
            y_pred = self.sigmoid(X.dot(self.weights)+self.bias)
            diff = y - y_pred

            dw = X.T.dot(diff) / n_samples
            db = np.sum(diff) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.where(self.sigmoid(X.dot(self.weights)+self.bias)<self.threshold, 1, 0)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
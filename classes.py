import numpy as np

class LinearRegression:
    def fit(self, X, y, learning_rate, iterations=100, L1_reg=0, L2_reg=0):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            diff = y_pred - y

            dw = np.dot(X.T, (diff)) / n_samples + L1_reg * np.sign(self.weights) + 2 * L2_reg * self.weights
            db = np.sum(diff) / n_samples

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def mse(self, y_true, y_pred):
        return np.mean((y_true-y_pred)**2)

    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true-y_pred))

    def mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true-y_pred) / y_true))

    def smape(self, y_true, y_pred):
        return np.mean(2*abs(y_true-y_pred)/(y_true+y_pred))

    def wape(self, y_true, y_pred):
        return np.sum(abs(y_true-y_pred))/np.sum(y_true)

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
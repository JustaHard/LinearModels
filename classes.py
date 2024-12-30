import numpy as np

class LinearModel:
    def __init__(self, learning_rate=0.01, iterations=1000, L1_reg=0, L2_reg=0):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0

        for _ in range(self.iterations):
            diff = self.calculate_diff(X, y)

            dw = X.T.dot(diff) / n_samples + self.L1_reg * np.sign(self.weights) + 2 * self.L2_reg * self.weights
            db = np.sum(diff) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

class LinearRegression(LinearModel):
    def calculate_diff(self, X, y):
        return np.dot(X, self.weights) + self.bias - y

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

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

class LogisticRegression(LinearModel):
    def __init__(self, threshold=0.5, learning_rate=0.01, iterations=1000, L1_reg=0, L2_reg=0):
        super().__init__(learning_rate, iterations, L1_reg, L2_reg)
        self.threshold = threshold

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calculate_diff(self, X, y):
        return y - self.sigmoid(X.dot(self.weights) + self.bias)

    def predict(self, X):
        return np.where(self.sigmoid(X.dot(self.weights)+self.bias)<self.threshold, 1, 0)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def confusion_matrix(self, y_true, y_pred):
        matrix = np.zeros((2, 2), dtype=int)

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
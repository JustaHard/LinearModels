import numpy as np

class LinearModel:
    """
    Абстрактный класс линейных моделей.
    """
    def fit(self, features:np.ndarray, targets:np.ndarray, *,
            learning_rate:float=0.01, iterations:int=1000, L1_reg:float=0,
            L2_reg:float=0, verbose:bool=False)->None:
        """
        Функция обучения модели.

        :param features: Матрица признаков (ndarray).
        :param targets: Матрица целевых значений (ndarray).
        :param learning_rate: Шаг градиентного спуска (float).
        :param iterations: Число итераций обучения модели (шагов градиентного спуска) (int > 0).
        :param L1_reg: Коэффициент L1 регуляризации (float).
        :param L2_reg: Коэффициент L2 регуляризации (float).
        :param verbose: Индикатор, определяющий вывод промежуточных результатов обучения (bool).
        """
        n_samples, n_features = features.shape
        self.weights, self.bias = np.zeros(n_features), 0

        for i in range(iterations):
            preds = self.predict(features)
            diff = preds - targets

            dw = features.T.dot(diff) / n_samples + L1_reg * np.sign(self.weights) + 2 * L2_reg * self.weights
            db = np.sum(diff) / n_samples

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            if verbose:
                print(f'Iteration number: {i+1}; Current loss: {np.mean(diff)}')

class LinearRegression(LinearModel):
    """
    Класс моделей линейной регрессии. Экземпляр абстрактного класса линейных моделей (LinearModel).
    """
    def predict(self, features:np.ndarray)->np.ndarray:
        """
        Расчитывает целевые значения по заданной матрице признаков на имеющихся весах.

        :param features: Матрица признаков значений, которые необходимо предсказать (ndarray).
        :return: Предсказанные целевые значения (ndarray).
        """
        return np.dot(features, self.weights) + self.bias

    def mse(self, features_true:np.ndarray, features_predicted:np.ndarray)->np.floating:
        """
        Расчитывает среднеквадратичную ошибку предсказанных значений относительно истинных целевых значений.

        :param features_true: Матрица истинных целевых значений (ndarray).
        :param features_predicted: Матрица предсказанных значений (ndarray).
        :return: Среднеквадратичная ошибка предсказанных значений относительно истинных целевых значений (floating).
        """
        return np.mean((features_true - features_predicted) ** 2)

    def mae(self, features_true:np.ndarray, features_predicted:np.ndarray)->np.floating:
        """
        Расчитывает среднюю абсолютную ошибку предсказанных значений относительно истинных целевых значений.

        :param features_true: Матрица истинных целевых значений (ndarray).
        :param features_predicted: Матрица предсказанных значений (ndarray).
        :return: Средняя абсолютная ошибка предсказанных
            значений относительно истинных целевых значений (floating).
        """
        return np.mean(np.abs(features_true - features_predicted))

    def mape(self, features_true:np.ndarray, features_predicted:np.ndarray)->np.floating:
        """
        Расчитывает среднюю абсолютную процентную ошибку предсказанных
            значений относительно истинных целевых значений.

        :param features_true: Матрица истинных целевых значений (ndarray).
        :param features_predicted: Матрица предсказанных значений (ndarray).
        :return: Средняя абсолютная процентная ошибка предсказанных
            значений относительно истинных целевых значений (floating).
        """
        return np.mean(np.abs((features_true - features_predicted) / features_true))

    def smape(self, features_true:np.ndarray, features_predicted:np.ndarray)->np.floating:
        """
        Расчитывает симметричную среднюю абсолютную процентную ошибку
            предсказанных значений относительно истинных целевых значений.

        :param features_true: Матрица истинных целевых значений (ndarray).
        :param features_predicted: Матрица предсказанных значений (ndarray).
        :return: Симметричная средняя абсолютная процентная ошибка
            предсказанных значений относительно истинных целевых значений (floating).
        """
        return np.mean(2 * abs(features_true - features_predicted) / (features_true + features_predicted))

    def wape(self, features_true:np.ndarray, features_predicted:np.ndarray)->np.floating:
        """
        Расчитывает взвешенную абсолютную процентную ошибку
            предсказанных значений относительно истинных целевых значений.

        :param features_true: Матрица истинных целевых значений (ndarray).
        :param features_predicted: Матрица предсказанных значений (ndarray).
        :return: Взвешенная абсолютная процентная ошибка предсказанных
            значений относительно истинных целевых значений (floating).
        """
        return np.sum(abs(features_true - features_predicted))/np.sum(features_true)

class LogisticRegression(LinearModel):
    """
    Класс моделей логистической регрессии (бинарной классификации).
    Экземпляр абстрактного класса линейных моделей (LinearModel)
    """
    def fit(self, features:np.ndarray, targets:np.ndarray, *,
            threshold:float=0.5, learning_rate:float=0.01, iterations:int=1000, L1_reg:float=0,
            L2_reg:float=0, verbose:bool=False)->None:
        """
        Функция обучения модели.

        :param features: Матрица признаков (ndarray).
        :param targets: Матрица целевых значений (ndarray).
        :param threshold: Граница, выше которой предсказанное целевое значение помечается, как 1 (float [0:1]).
        :param learning_rate: Шаг градиентного спуска (float).
        :param iterations: Число итераций обучения модели (шагов градиентного спуска) (int > 0).
        :param L1_reg: Коэффициент L1 регуляризации (float).
        :param L2_reg: Коэффициент L2 регуляризации (float).
        :param verbose: Индикатор, определяющий вывод промежуточных результатов обучения (bool).
        """
        self.threshold = threshold
        super().fit(features, targets, learning_rate=learning_rate, iterations=iterations,
                    L1_reg=L1_reg, L2_reg=L2_reg, verbose=verbose)

    def predict(self, X):
        self.preds = self.sigmoid(X.dot(self.weights) + self.bias)
        if isinstance(self, LinearModel) and not isinstance(self, LogisticRegression):
            return self.preds
        else:
            return np.where(self.preds < self.threshold, 0, 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def find_TP(self, y_true, y_pred):
        return np.sum((y_pred==1) & (y_true==1))

    def find_FP(self, y_true, y_pred):
        return np.sum((y_pred==1) & (y_true!=1))

    def find_FN(self, y_true, y_pred):
        return np.sum((y_pred!=1) & (y_true==1))

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def confusion_matrix(self, y_true, y_pred):
        matrix = np.zeros((2, 2), dtype=int)

        for true, pred in zip(y_true, y_pred):
            matrix[true, pred] += 1

        return matrix

    def precision(self, y_true, y_pred):
        TP = self.find_TP(y_true, y_pred)
        FP = self.find_FP(y_true, y_pred)

        if TP + FP == 0:
            return 0.0
        return TP / (TP + FP)

    def recall(self, y_true, y_pred):
        TP = self.find_TP(y_true, y_pred)
        FN = self.find_FN(y_true, y_pred)

        if TP + FN == 0:
            return 0.0
        return TP / (TP + FN)

    def f1(self, y_true, y_pred):
        recall = self.recall(y_true, y_pred)
        precision = self.precision(y_true, y_pred)

        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    def f_beta(self, y_true, y_pred, beta):
        recall = self.recall(y_true, y_pred)
        precision = self.precision(y_true, y_pred)

        if recall + precision == 0:
            return 0.0
        return (beta ** 2 + 1) * recall * precision/(recall + beta ** 2 * precision)
import numpy as np
import pandas as pd


class LinearModel:
    """
    Абстрактный класс линейных моделей.
    """
    def fit(self, features:np.ndarray[int, float]|pd.DataFrame[int, float],
            targets:np.ndarray[int, float]|pd.Series[int, float],
            *,
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
    def predict(self, features:np.ndarray[int, float]|pd.DataFrame[int, float])->np.ndarray[int, float]:
        """
        Расчитывает целевые значения по заданной матрице признаков на имеющихся весах.

        :param features: Матрица признаков значений, которые необходимо предсказать (ndarray).
        :return: Предсказанные целевые значения (ndarray).
        """
        return np.dot(features, self.weights) + self.bias

    def mse(self, features_true:np.ndarray[int, float]|pd.DataFrame[int, float],
            features_predicted:np.ndarray[int, float]|pd.DataFrame[int, float])->np.floating:
        """
        Расчитывает среднеквадратичную ошибку предсказанных значений относительно истинных целевых значений.

        :param features_true: Матрица истинных целевых значений (ndarray).
        :param features_predicted: Матрица предсказанных значений (ndarray).
        :return: Среднеквадратичная ошибка предсказанных значений относительно истинных целевых значений (floating).
        """
        return np.mean((features_true - features_predicted) ** 2)

    def mae(self, features_true:np.ndarray[int, float]|pd.DataFrame[int, float],
            features_predicted:np.ndarray[int, float]|pd.DataFrame[int, float])->np.floating:
        """
        Расчитывает среднюю абсолютную ошибку предсказанных значений относительно истинных целевых значений.

        :param features_true: Матрица истинных целевых значений (ndarray).
        :param features_predicted: Матрица предсказанных значений (ndarray).
        :return: Средняя абсолютная ошибка предсказанных
            значений относительно истинных целевых значений (floating).
        """
        return np.mean(np.abs(features_true - features_predicted))

    def mape(self, features_true:np.ndarray[int, float]|pd.DataFrame[int, float],
             features_predicted:np.ndarray[int, float]|pd.DataFrame[int, float])->np.floating:
        """
        Расчитывает среднюю абсолютную процентную ошибку предсказанных
            значений относительно истинных целевых значений.

        :param features_true: Матрица истинных целевых значений (ndarray).
        :param features_predicted: Матрица предсказанных значений (ndarray).
        :return: Средняя абсолютная процентная ошибка предсказанных
            значений относительно истинных целевых значений (floating).
        """
        return np.mean(np.abs((features_true - features_predicted) / features_true))

    def smape(self, features_true:np.ndarray[int, float]|pd.DataFrame[int, float],
              features_predicted:np.ndarray[int, float]|pd.DataFrame[int, float])->np.floating:
        """
        Расчитывает симметричную среднюю абсолютную процентную ошибку
            предсказанных значений относительно истинных целевых значений.

        :param features_true: Матрица истинных целевых значений (ndarray).
        :param features_predicted: Матрица предсказанных значений (ndarray).
        :return: Симметричная средняя абсолютная процентная ошибка
            предсказанных значений относительно истинных целевых значений (floating).
        """
        return np.mean(2 * abs(features_true - features_predicted) / (features_true + features_predicted))

    def wape(self, features_true:np.ndarray[int, float]|pd.DataFrame[int, float],
             features_predicted:np.ndarray[int, float]|pd.DataFrame[int, float])->np.floating:
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
    def fit(self, features:np.ndarray[int, float]|pd.DataFrame[int, float],
            targets:np.ndarray[int]|pd.DataFrame[int],
            *,
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

    def predict(self, features:np.ndarray[int, float]|pd.DataFrame[int, float])->np.ndarray[int, float]:
        """
        Рассчитывает целевые значения по заданной матрице признаков на имеющихся весах

        :param features: Матрица признаков значений, которые необходимо предсказать (ndarray).
        :return: Предсказанные целевые значения (ndarray).
        """
        self.preds = self.sigmoid(features.dot(self.weights) + self.bias)
        if isinstance(self, LinearModel) and not isinstance(self, LogisticRegression):
            return self.preds
        else:
            return np.where(self.preds < self.threshold, 0, 1)

    def sigmoid(self, values:np.ndarray[int, float])->np.ndarray[int, float]:
        """
        Приводит заданные значения в формат от 0 до 1.

        :param values: Матрица значений, которые необходимо изменить (ndarray).
        :return: Матрица значений в формате от 0 до 1 (ndarray).
        """
        return 1 / (1 + np.exp(-values))

    def find_TP(self, targets_true:np.ndarray[int]|pd.DataFrame[int],
                targets_predicted:np.ndarray[int]|pd.DataFrame[int])->np.integer:
        """
        Расчитывает количество верноопределенных положительных целевых значений (1).

        :param targets_true: Матрица истинных целевых значений (ndarray).
        :param targets_predicted: Матрица предсказанных целевых значений (ndarray).
        :return: Количество верноопределенных положительных значений (integer).
        """
        return np.sum((targets_predicted == 1) & (targets_true == 1))

    def find_FP(self, targets_true:np.ndarray[int]|pd.DataFrame[int],
                targets_predicted:np.ndarray[int]|pd.DataFrame[int])->np.integer:
        """
        Расчитывает количество значений, определенных, как положительные (1), когда их истинное значение отрицательно(0).

        :param targets_true: Матрица истинных целевых значений (ndarray).
        :param targets_predicted: Матрица предсказанных целевых значений (ndarray).
        :return: Количество значений, определенных, как положительные, когда их истинное значение отрицательно (integer).
        """
        return np.sum((targets_predicted == 1) & (targets_true != 1))

    def find_FN(self, targets_true:np.ndarray[int]|pd.DataFrame[int],
                targets_predicted:np.ndarray[int]|pd.DataFrame[int])->np.integer:
        """
        Расчитывает количество значений, определенных, как отрицательные (0), когда их истинное значение положительно (1).

        :param targets_true: Матрица истинных целевых значений (ndarray).
        :param targets_predicted: Матрица предсказанных целевых значений (ndarray).
        :return: Количество значений, определенных, как отрицательные, когда их истинное значение положительно (integer).
        """
        return np.sum((targets_predicted != 1) & (targets_true == 1))

    def accuracy(self, targets_true:np.ndarray[int]|pd.DataFrame[int],
                 targets_predicted:np.ndarray[int]|pd.DataFrame[int])->np.floating:
        """
        Расчитывает долю правильных прогнозов по отношению к общему количеству предположений.

        :param targets_true: Матрица истинных целевых значений (ndarray).
        :param targets_predicted: Матрица предсказанных целевых значений (ndarray).
        :return: Доля правильных прогнозов по отношению к общему количеству предположений (floating).
        """
        return np.mean(targets_true == targets_predicted)

    def confusion_matrix(self, targets_true:np.ndarray[int]|pd.DataFrame[int],
                         targets_predicted:np.ndarray[int]|pd.DataFrame[int])->np.ndarray[int]:
        """
        Создает матрицу ошибок предсказаний модели.

        :param targets_true: Матрица истинных целевых значений (ndarray).
        :param targets_predicted: Матрица предсказанных целевых значений (ndarray).
        :return: Матрица ошибок предсказаний модели (ndarray).
        """
        matrix = np.zeros((2, 2), dtype=int)

        for true, pred in zip(targets_true, targets_predicted):
            matrix[true, pred] += 1

        return matrix

    def precision(self, targets_true:np.ndarray[int]|pd.DataFrame[int],
                  targets_predicted:np.ndarray[int]|pd.DataFrame[int])->np.floating:
        """
        Расчитывает долю правильно предсказанных положительных объектов среди всех объектов, предсказанных положительным классом.

        :param targets_true: Матрица истинных целевых значений (ndarray).
        :param targets_predicted: Матрица предсказанных целевых значений (ndarray).
        :return: Доля правильно предсказанных положительных объектов среди всех объектов, предсказанных положительным классом (floating|float).
        """
        TP = self.find_TP(targets_true, targets_predicted)
        FP = self.find_FP(targets_true, targets_predicted)

        if TP + FP == 0:
            return np.floating(0.0)
        return TP / (TP + FP)

    def recall(self, targets_true:np.ndarray[int]|pd.DataFrame[int],
               targets_predicted:np.ndarray[int]|pd.DataFrame[int])->np.floating:
        """
        Расчитывает долю правильно предсказанных положительных объектов среди всех объектов положительного класса

        :param targets_true: Матрица истинных целевых значений (ndarray).
        :param targets_predicted: Матрица предсказанных целевых значений (ndarray).
        :return: Доля правильно предсказанных положительных объектов среди всех объектов положительного класса (floating|float).
        """
        TP = self.find_TP(targets_true, targets_predicted)
        FN = self.find_FN(targets_true, targets_predicted)

        if TP + FN == 0:
            return np.floating(0.0)
        return TP / (TP + FN)

    def f1(self, targets_true:np.ndarray[int]|pd.DataFrame[int],
           targets_predicted:np.ndarray[int]|pd.DataFrame[int])->np.floating:
        """
        Расчитывает среднее гармоническое метрик Precision и Recall при их равной важности.

        :param targets_true: Матрица истинных целевых значений (ndarray).
        :param targets_predicted: Матрица предсказанных целевых значений (ndarray).
        :return: Среднее гармоническое метрик Precision и Recall при их равной важности
        """
        recall = self.recall(targets_true, targets_predicted)
        precision = self.precision(targets_true, targets_predicted)

        if recall + precision == 0:
            return np.floating(0.0)
        return 2 * recall * precision / (recall + precision)

    def f_beta(self, targets_true:np.ndarray[int]|pd.DataFrame[int],
               targets_predicted:np.ndarray[int]|pd.DataFrame[int],
               *,
               beta:float=1.0)->np.floating:
        """
        Расчитывает среднее гармоническое метрик Precision и Recall при их разной важности.

        :param targets_true: Матрица истинных целевых значений (ndarray).
        :param targets_predicted: Матрица предсказанных целевых значений (ndarray).
        :param beta: Коэффициент важности Precision: beta = 1 - Равная важность коэффициентов;
            beta > 1 - повышенная важность Precision; 0 < beta < 1 - повышенная важность Recall (float).
        :return: Среднее гармоническое метрик Precision и Recall при их разной важности (floating|float).
        """
        recall = self.recall(targets_true, targets_predicted)
        precision = self.precision(targets_true, targets_predicted)

        if recall + precision == 0:
            return np.floating(0.0)
        return (beta ** 2 + 1) * recall * precision/(recall + beta ** 2 * precision)
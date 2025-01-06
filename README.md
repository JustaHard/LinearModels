# Linear Models
## Описание
Linear Models - это реализация решения задач машинного обучения 
с помощью линейных моделей, а именно:
* Реализация линейной регрессии для решения задач предсказания 
непрерывных величин.
* Реализация бинарной логистической регрессии 
(линейной классификации) для решения задач предсказания одного 
из двух классов, к которому принадлежит тот или иной объект выборки.
## Установка
1. Клонируйте репозиторий:
    ```bash
   git clone https://github.com/JustaHard/LinearModels.git
2. Перейдите в директорию проекта:
    ```bash
   cd LinearModels
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt

## Использование
1. Импортируйте необходимые функции из файла `classes.py`
   ```Python
   from classes import *
   ```
2. Создайте экземпляр неоходимого вам класса в зависимости 
от типа решаемой задачи - регрессия или классификация
   ```Python
   model = LinearRegression()
   ```
   или
   ```Python
   model = LogisticRegression()
   ```    
3. Обучите модель на тренировочном датасете
   ```Python
   model.fit(X_train, y_train)
   ```
4. Используйте обученную модель для предсказания значений 
или классов
   ```Python
   y_preds = model.predict(X_test)
   ```
5. Вы можете оценивать эффективность работы модели с 
помощью различных метрик:
   1. Для линейной регрессии:
      * Средняя квадратичная ошибка - `model.mse()`
      * Средняя абсолютная ошибка - `model.mae()`
      * Средняя абсолютная процентная ошибка - `model.mape()`
      * Симметричная средняя абсолютная ошбика - `model.smape()`
      * Взвешенная абсолютная процентная ошибка - `model.wape()`
   2. Для логистической регрессии:
      * Точность модели - `model.accuracy()`
      * Матрица ошибок предсказаний модели - `model.confusion_matrix()`
      * Доля правильно предсказанных положительных 
      объектов среди всех объектов, предсказанных положительным 
      классом (Precision) - `model.precision()`
      * Доля правильно предсказанных положительных объектов среди
      всех положительных объектов (Recall) - `model.recall()`
      * Среднее гармоническое между Precision и Recall при их равной
      важности (f1 score) - `model.f1()`
      * Среднее гармоническое между Precision и Recall при их разной
      важности - `model.f_beta()`

Узнать подробнее про каждую функцию и ее параметры можно с помощью 
документации самой функции
## Пример кода
Пример кода представлен в файле `example.ipynb`
## Требования
Перечень необходимых для работы библиотек представлен в файле requirements.txt
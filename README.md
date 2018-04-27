# Отчет по лабораторной работе
## по курсу "Искусственый интеллект"

### Выполнил:
    Симахин Иван
## Задание

Построить классификатор выявляющий плохие
ответы пользователя на языке Python.

В качестве дата сета был выбран набор ответов для языка R

## Отчет по ходу работы

## Используемый алгоритм
В качестве классификатора была взята логистическая регрессия.

![image](Logistic-curve.png)

Где в качестве функции используется сигмойда

![sigmoid](sigmoid.svg)

Идея работы логистической регресси в том, чтобы получить на вход число(качество объекта)
и выдать вероятность того, что объект принадлежит классу.

## Ход работы

Вначале происходит загрузка данных из csv файла
```python
def __init__(self, path, min_goodness: int):
    self.min_goodness = min_goodness
    self.df = pandas.read_csv(path)
```

Первая часть важных преобразования происходит в методе calculate.
Происходит разбиение на две выборки обучающую/тестовую, обработка тестовой выборки.
Для каждой строки из датасета берется поле Body(тело ответа) и удаляются html тэги с помощью
метода transform
```python 
def transform(self, a: str):
    a = re.sub(r"(</.*>)", "", a)
    a = re.sub(r"(<.*>)", "", a)
    return a
```

Преобразованную строку добавляю в список преобразованных строк, так же формируется список
"полезности" ответа, если рейтинг ответа больше заданного числа, то ответ считается полезным.

```python
for index, row in train_df.iterrows():
    if row["Score"] > self.min_goodness:
        train_goodness.append(1)
    else:
        train_goodness.append(0)
    train_bodies.append(self.transform(row["Body"]))
```

Полученные данные отправляются в модель 

```python
self.classifier = LogisticRegression()
        self.classifier.fit(X_train, train_goodness)
```

После чего такие же преобразование выполняются для тестовой выборки(без создания новой модели).
Преобразованная выборка отправляется в модель, после чего сравнивается количество предсказанных значений
с реальными даннами и выводится точность
```python
def predict(self, test_df):
    test_bodies = []
    test_goodness = []
    for index, row in test_df.iterrows():
        test_bodies.append(self.transform(row["Body"]))
        if row["Score"] > self.min_goodness:
            test_goodness.append(1)
        else:
            test_goodness.append(0)
    X_test = self.vectorizer.transform(test_bodies)
    x_predict = self.classifier.predict(X_test)
    cnt = 0
    for i in range(0, len(x_predict)):
        if test_goodness[i] == x_predict[i]:
            cnt += 1
    print("Predicted values : ", cnt / len(test_goodness), sep=" ")
    return cnt / len(test_goodness)
```

Для тестового примера я взял значение полезности в 2 и разделение выборок как 3:1 , точность составила
80%
"""
Машинное обучение

Есть набор точек на вход, нужно получить функцию,
которая будет давать результат соответсвующий ожидаемому.

Эта функция должна - улавливать важные сигналы
                   - игнорировать помехи
                   - хорошо работать на новых неизвестных данных

Нет информации о функции, которая дала данные

Проблемы:
        Функций, удовлетворяющих условию вход-выход много
        Сложно оценить правильность работы на новых данных


        

Источник знаний ---> Data #1 ---> Обучение ---> Model #1
            |          |                 -----------|
            |          |                 |          V
            |          |----------------------> Предсказание ---> Результат #1
            |                            |
            |                            V
            |------> Data #2 ---> Предсказание -----------------> Результат #2

Выбор данный имеет большре значение!
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = sns.load_dataset("iris")
print(iris.head())

#print(type(iris.values))
#
#print(iris.values.shape)
#
#print(iris.columns)
#
#print(iris.index)
#
#sns.pairplot(iris, hue="species")
#
#plt.savefig(plt.png)

# Строки - образцы - отдельный объект (sample)
# Столбцы - признаки (feature)
# Матрица признаков [число образцов на число признаков] - признаки - Независимая переменная
# Целевой массив (target, label) [1 на число образцов] - зависимая переменная

#_iris=iris.drop("species", axis=1)
#rint(X_iris)
#
#_iris=iris["species"]
#rint(y_iris)

# 1. Выбирается класс модели
# 2. Выбирается гипперпараметры модели
# 3. На основе данных создаётся матрица признаков и целевой вектор
# 4. Обучение модели fit()
# 5. Обученная модель применяется к новым данным
#   5.1. Обучение с учителем - predict()
#   5.2 Обучение без учителя - predict() или transform()

# C учителем. Регрессия. Линейная регрессия

x = iris[iris["species"]=="setosa"].iloc[:, 0].to_numpy()
y = iris[iris["species"]=="setosa"].iloc[:, 1].to_numpy()

print(type(x))

# 1. Выбирается класс модели
from sklearn.linear_model import LinearRegression

# 2. Выбирается гипперпараметры модели
model=LinearRegression()

# 3. На основе данных создаётся матрица признаков и целевой вектор


# 4. Обучение модели fit()
reg=model.fit(x[:, np.newaxis], y)

plt.scatter(x, y)


# 5. Обученная модель применяется к новым данным
#   5.1. Обучение с учителем - predict()

xfit = np.linspace(0, x.max(), 1000)
yfit = model.predict(xfit[:, None])

plt.plot(xfit, yfit, "r")

plt.plot(xfit, xfit * reg.coef_ + reg.intercept_, "k")

# y = kx + b



from sklearn.pipeline import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(7), LinearRegression())
reg = model.fit(x[:, np.newaxis], y)

xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict(xfit[:, None])

plt.scatter(x, y)
plt.plot(xfit, yfit, "r")

# Классификация. Логистическая регрессия

x_0 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y_0 = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()
x_1 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()
y_1 = iris[iris["species"] == "versicolor"].iloc[:, 1].to_numpy()

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

x_00 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
x_11 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()

# plt.scatter(x_0, y_0, color="red", alpha=0.5)

plt.scatter(x_00, np.full(50, 1), color="red", alpha=0.5)
plt.scatter(x_11, np.full(50, 5), color="green", alpha=0.5)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

x = iris[iris["species"] != "virginica"].iloc[:, 0].to_numpy()
print(x.shape)
y = iris[iris["species"] != "virginica"].iloc[:, 1]
print(y.shape)

model.fit(x[:, None], y)

xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict(xfit[:, None])

print(yfit)
plt.plot(xfit, yfit[:, 1], "red")

plt.plot(xfit, 1+4*yfit[:, 1], "red")
plt.plot(xfit, 1+4*yfit[:, 0], "blue")

# Деревья решений
from sklearn.linear_model import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(x, y)

np.meshgrid(
    np.linspace(x[:, :0].min(), x[:, :0].max(), 1000),
    np.linspace(x[:, 1:1].min(), x[:, 1:1].max(), 1000)
)

print(np.c_[[1,2,3,4,5], [10,20,30,40,50]])

print(np.ravel([[1,2,3,4,5], [10,20,30,40,50]]))

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)

Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]) # .reshape(xx.shape)

ax = plt.gca()

ax.contourf(xx, yy, Z, alpha=0.3)


plt.savefig("plt.png")

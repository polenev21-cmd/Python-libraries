import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Загрузка данных
iris = load_iris()
X_iris, Y_iris = iris.data, iris.target

"""
X_iris [Длина чашелистника, Ширина чашелистника, Длина лепестка, Ширина лепестка]
Y_iris [0 - Ирис щетинистый, 1 - Ирис разноцветный, 2 - Ирис виргинский]
"""

# Берём только два класса (0 и 1)
X_filtered = X_iris[(Y_iris == 0) | (Y_iris == 1)]
Y_filtered = Y_iris[(Y_iris == 0) | (Y_iris == 1)]

# Разделение на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(
    X_filtered, Y_filtered, test_size=0.3, random_state=0
)

# Обучение модели случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, Y_train)

# Предсказание на тестовой выборке (для метрик)
y_pred = model.predict(X_test)

# Вывод точности
from sklearn.metrics import accuracy_score
print(f"Точность на тесте: {accuracy_score(Y_test, y_pred):.2f}")

# Визуализация разделяющей границы по первым двум признакам
xx, yy = np.meshgrid(
    np.linspace(X_filtered[:, 0].min(), X_filtered[:, 0].max(), 100),
    np.linspace(X_filtered[:, 1].min(), X_filtered[:, 1].max(), 100),
)

# Предсказываем класс для всех точек сетки
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Рисуем фон разделяющей границы
plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])

# Исправленная часть: отображаем реальные точки из ОБУЧАЮЩЕЙ выборки по их ИСТИННЫМ классам
X0 = X_filtered[Y_filtered == 0][:, 0]
Y0 = X_filtered[Y_filtered == 0][:, 1]

X1 = X_filtered[Y_filtered == 1][:, 0]
Y1 = X_filtered[Y_filtered == 1][:, 1]

plt.scatter(X0, Y0, color="red", edgecolor='k', label=iris.target_names[0])
plt.scatter(X1, Y1, color="blue", edgecolor='k', label=iris.target_names[1])

plt.xlabel("Длина чашелистника (см)")
plt.ylabel("Ширина чашелистника (см)")
plt.title("Разделяющая граница Random Forest (классы 0 и 1)")
plt.legend()
plt.show()
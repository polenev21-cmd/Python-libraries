import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка и подготовка данных
iris = load_iris()
X_iris, Y_iris = iris.data, iris.target

# Выбираем два сорта (Setosa и Versicolor)
mask = (Y_iris == 0) | (Y_iris == 1)
X_filtered = X_iris[mask]
Y_filtered = Y_iris[mask]

# Масштабирование (опционально, но полезно)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y_filtered, test_size=0.3, random_state=42
)

# Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Предсказания
y_pred = rf.predict(X_test)
accuracy = rf.score(X_test, y_test)

# Визуализация важности признаков
feature_names = ['Длина чашелистика', 'Ширина чашелистика', 
                  'Длина лепестка', 'Ширина лепестка']

plt.figure(figsize=(10, 5))

# График важности признаков
plt.subplot(1, 2, 1)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(4), importances[indices])
plt.xticks(range(4), [feature_names[i] for i in indices], rotation=45)
plt.title('Важность признаков (RandomForest)')

# График: два лучших признака (длина и ширина лепестка)
plt.subplot(1, 2, 2)
plt.scatter(X_filtered[Y_filtered==0, 2], X_filtered[Y_filtered==0, 3],
           c='red', label='Setosa', alpha=0.7)
plt.scatter(X_filtered[Y_filtered==1, 2], X_filtered[Y_filtered==1, 3],
           c='blue', label='Versicolor', alpha=0.7)
plt.xlabel('Длина лепестка (см)')
plt.ylabel('Ширина лепестка (см)')
plt.title(f'RandomForest: точность = {accuracy:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Точность RandomForest: {accuracy:.3f}")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix

# Загрузка данных
iris = load_iris()
X = iris.data
y = iris.target

# Выбираем только два класса: 0 (setosa) и 1 (versicolor)
mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

print("Выбранные классы:", iris.target_names[[0, 1]])
print("Размер выборки:", X.shape)
print("Количество признаков:", X.shape[1])
print()

# Применяем KMeans из sklearn
kmeans = KMeans(n_clusters=2,           # количество кластеров
                random_state=42,        # фиксируем случайность для воспроизводимости
                n_init=10,              # количество запусков с разными центроидами
                max_iter=300)           # максимальное количество итераций

# Обучаем модель
kmeans.fit(X)

# Получаем метки кластеров
y_kmeans = kmeans.labels_

# Сопоставляем метки кластеров с истинными классами
# KMeans может назначить кластеру номер 0 или 1 в любом порядке
acc_original = accuracy_score(y, y_kmeans)
acc_inverted = accuracy_score(y, 1 - y_kmeans)

if acc_inverted > acc_original:
    y_kmeans = 1 - y_kmeans
    acc = acc_inverted
    print("Метки кластеров были инвертированы для сопоставления")
else:
    acc = acc_original

print("\n=== Результаты KMeans (sklearn) ===")
print(f"Точность сопоставления с истинными метками: {acc:.4f} ({acc*100:.1f}%)")
print(f"Количество итераций: {kmeans.n_iter_}")
print(f"Сошёлся ли алгоритм: {kmeans.n_iter_ < kmeans.max_iter}")

# Матрица ошибок
print("\nМатрица ошибок:")
cm = confusion_matrix(y, y_kmeans)
print(cm)
print("\nРасшифровка матрицы ошибок:")
print(f"  Верно определено Setosa:        {cm[0,0]}")
print(f"  Setosa ошибочно как Versicolor: {cm[0,1]}")
print(f"  Верно определено Versicolor:    {cm[1,1]}")
print(f"  Versicolor ошибочно как Setosa: {cm[1,0]}")

# Центроиды
print("\nКоординаты центроидов кластеров:")
print(f"  Кластер 0: {kmeans.cluster_centers_[0]}")
print(f"  Кластер 1: {kmeans.cluster_centers_[1]}")

# Сумма квадратов расстояний до центроидов (inertia)
print(f"\nInertia (сумма квадратов расстояний): {kmeans.inertia_:.2f}")

# Визуализация результатов
plt.figure(figsize=(15, 5))

# График 1: Истинные классы
plt.subplot(1, 3, 1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Setosa', alpha=0.7, edgecolors='black')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Versicolor', alpha=0.7, edgecolors='black')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Истинные классы')
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Результат KMeans
plt.subplot(1, 3, 2)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c='lightblue', label='KMeans кластер 0', alpha=0.7, edgecolors='black')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c='salmon', label='KMeans кластер 1', alpha=0.7, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='X', s=250, c='green', label='Центроиды', edgecolors='black', linewidth=2)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('KMeans кластеры')
plt.legend()
plt.grid(True, alpha=0.3)

# График 3: Ошибки кластеризации
plt.subplot(1, 3, 3)
correct = (y == y_kmeans)
wrong = (y != y_kmeans)
plt.scatter(X[correct, 0], X[correct, 1], c='green', label=f'Верно ({sum(correct)})', alpha=0.7, marker='o', s=50)
plt.scatter(X[wrong, 0], X[wrong, 1], c='red', label=f'Ошибка ({sum(wrong)})', alpha=0.7, marker='x', s=100, linewidth=2)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title(f'Ошибки кластеризации (точность: {acc*100:.1f}%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Дополнительная визуализация в пространстве PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='blue', label='Setosa', alpha=0.7)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='red', label='Versicolor', alpha=0.7)
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('Истинные классы (PCA)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_pca[y_kmeans == 0, 0], X_pca[y_kmeans == 0, 1], c='lightblue', label='KMeans кластер 0', alpha=0.7)
plt.scatter(X_pca[y_kmeans == 1, 0], X_pca[y_kmeans == 1, 1], c='salmon', label='KMeans кластер 1', alpha=0.7)
# Центроиды в PCA пространстве
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=250, c='green', label='Центроиды', edgecolors='black')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('KMeans кластеры (PCA)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Анализ качества кластеризации
print("\n=== Дополнительные метрики качества ===")
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

silhouette = silhouette_score(X, y_kmeans)
calinski = calinski_harabasz_score(X, y_kmeans)
davies_bouldin = davies_bouldin_score(X, y_kmeans)

print(f"Силуэтный коэффициент: {silhouette:.4f}")
print(f"  (ближе к 1 → лучше разделение)")
print(f"Индекс Калински-Харабаса: {calinski:.2f}")
print(f"  (чем выше, тем лучше)")
print(f"Индекс Дэвиса-Болдина: {davies_bouldin:.4f}")
print(f"  (чем ниже, тем лучше)")
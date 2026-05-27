import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris=load_iris()
X_iris, Y_iris=iris.data, iris.target

"""
X_iris [Длина чашелистника, Ширина чашелистника, Длина лепестка, Ширина лепестка]
Y_iris [0 - Ирис щетинистый, 1 - Ирис разноцветный, 2 - Ирис виргинский]
"""

X_filtered=X_iris[(Y_iris==0) | (Y_iris==1)]
Y_filtered=Y_iris[(Y_iris==0) | (Y_iris==1)]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_filtered)

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)

X_setosa_pca=X_pca[Y_filtered==0]
X_virginica_pca=X_pca[Y_filtered==1]

plt.figure(figsize=(10, 5))

plt.scatter(X_setosa_pca[:, 0], X_setosa_pca[:, 1], c='blue', label='Setosa', alpha=0.5, s=60)

plt.scatter(X_virginica_pca[:, 0], X_virginica_pca[:, 1], c='red', label='Versicolor', alpha=0.5, s=60)

plt.xlabel(f'Первая главная компонента (PC1) - {pca.explained_variance_ratio_[0]*100:.1f}%')
plt.ylabel(f'Вторая главная компонента (PC2) - {pca.explained_variance_ratio_[1]*100:.1f}%')
plt.title('PCA проекция классов Setosa и Versicolor')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

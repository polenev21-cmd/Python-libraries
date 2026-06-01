import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

iris=load_iris()
X_iris, Y_iris=iris.data, iris.target

"""
X_iris [Длина чашелистника, Ширина чашелистника, Длина лепестка, Ширина лепестка]
Y_iris [0 - Ирис щетинистый, 1 - Ирис разноцветный, 2 - Ирис виргинский]
"""

X_filtered=X_iris[Y_iris!=2][:, 0:2]
Y_filtered=Y_iris[Y_iris!=2]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_filtered)

model=KMeans(n_clusters=2)
model.fit(X_scaled)

xx, yy=np.meshgrid(
    np.linspace(X_scaled[:, 0].min()*1.05, X_scaled[:, 0].max()*1.05, 100),
    np.linspace(X_scaled[:, 1].min()*1.05, X_scaled[:, 1].max()*1.05, 100))

x_0=X_scaled[Y_filtered==0][:, 0]
y_0=X_scaled[Y_filtered==0][:, 1]
x_1=X_scaled[Y_filtered==1][:, 0]
y_1=X_scaled[Y_filtered==1][:, 1]

Z=model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])
plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="blue", alpha=0.5)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], size=100, color='black', marker='X')
plt.savefig("KMeans.jpg")
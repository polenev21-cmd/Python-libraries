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

X_filtered=X_iris[Y_iris!=2]
Y_filtered=Y_iris[Y_iris!=2]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_filtered)

model=PCA(n_components=2)
x=model.fit_transform(X_scaled)

x_0=x[Y_filtered==0][:,0]
y_0=x[Y_filtered==0][:,1]
x_1=x[Y_filtered==1][:,0]
y_1=x[Y_filtered==1][:,1]

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="blue", alpha=0.5)
plt.savefig("PCA.jpg")



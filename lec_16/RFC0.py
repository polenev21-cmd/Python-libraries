import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


iris=load_iris()
X_iris, Y_iris=iris.data, iris.target

"""
X_iris [Длина чашелистника, Ширина чашелистника, Длина лепестка, Ширина лепестка]
Y_iris [0 - Ирис щетинистый, 1 - Ирис разноцветный, 2 - Ирис виргинский]
"""

X_filtered=X_iris[(Y_iris==0) | (Y_iris==1)]
Y_filtered=Y_iris[(Y_iris==0) | (Y_iris==1)]

#scaler=StandardScaler()
#X_scaled=scaler.fit_transform(X_filtered)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_filtered, Y_filtered, test_size=0.3, random_state=0
)

model=RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, Y_train)

#xx, yy = np.meshgrid(
#    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
#    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
#)

x_0=x[y==0][:,0]
y_0=x[y==0][:,1]

x_1=x[y==1][:,0]
y_1=x[y==1][:,1]

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="blue", alpha=0.5)

Z=model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax = plt.gca()
ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5])
plt.show()

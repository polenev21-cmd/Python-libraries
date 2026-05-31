import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

iris=load_iris()
X_iris, Y_iris=iris.data, iris.target

"""
X_iris [Длина чашелистника, Ширина чашелистника, Длина лепестка, Ширина лепестка]
Y_iris [0 - Ирис щетинистый, 1 - Ирис разноцветный, 2 - Ирис виргинский]
"""

X_filtered=X_iris[Y_iris!=2][:, 0:2]
Y_filtered=Y_iris[Y_iris!=2]

X_train, X_test, Y_train, Y_test=train_test_split(
    X_filtered, Y_filtered, test_size=0.3, random_state=0
)

model=RandomForestClassifier()
model.fit(X_train, Y_train)
y_pred=model.predict(X_test)

xx, yy=np.meshgrid(
    np.linspace(X_filtered[:, 0].min()*0.95, X_filtered[:, 0].max()*1.05, 100),
    np.linspace(X_filtered[:, 1].min()*0.95, X_filtered[:, 1].max()*1.05, 100),
)

x_0=X_filtered[Y_filtered==0][:,0]
y_0=X_filtered[Y_filtered==0][:,1]
x_1=X_filtered[Y_filtered==1][:,0]
y_1=X_filtered[Y_filtered==1][:,1]

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])
plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="blue", alpha=0.5)
plt.savefig("RFC.jpg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
Y=iris.target

"""
X_iris [Длина чашелистника, Ширина чашелистника, Длина лепестка, Ширина лепестка]
Y_iris [0 - Ирис щетинистый, 1 - Ирис разноцветный, 2 - Ирис виргинский]
"""

data=X[:,[0, 2]]

data_of_setosa=data[Y==0]
data_of_versicolor=data[Y==1]
data_of_verginica=data[Y==2]


#plt.scatter(data_of_setosa[:,0], data_of_setosa[:,1])
#plt.scatter(data_of_versicolor[:,0], data_of_versicolor[:,1])

X=data[(Y==0)|(Y==1)]
Y=Y[(Y==0)|(Y==1)]


from sklearn.tree import DecisionTreeClassifier
#
#model=DecisionTreeClassifier()
#model.fit(X,Y)
#
x1_p=np.linspace(min(data[:,0]), max(data[:,0]))
x2_p=np.linspace(min(data[:,1]), max(data[:,1]))

X1_p, X2_p=np.meshgrid(x1_p, x2_p)

X_p = np.vstack([X1_p.ravel(), X2_p.ravel()]).T
#
#Y_p=model.predict(X_p)
#
#plt.contourf(
#    X1_p,
#    X2_p,
#    Y_p.reshape(X1_p.shape),
#    alpha=0.3,
#    levels=[-0.5, 0.5, 1.5]
#)





#max_depth = [[1, 2, 3, 4], [5, 6, 7, 8]]
#
#fig, ax = plt.subplots(2, 4, sharex="col", sharey="row", figsize=(12, 6))
#
#for i in range(2):
#    j = 0
#    for md in max_depth[i]:
#        model = DecisionTreeClassifier(max_depth=md)
#        model.fit(X, Y)
#        Y_p = model.predict(X_p)
#        ax[i,j].scatter(data_of_setosa[:,0], data_of_setosa[:,1])
#        ax[i,j].scatter(data_of_versicolor[:,0], data_of_versicolor[:,1])
#        ax[i,j].contourf(X1_p, X2_p, Y_p.reshape(X1_p.shape), alpha=0.3)
#        j+=1
#plt.show()




X=iris.data
Y=iris.target
data_of_versicolor_A=data_of_versicolor[:25, :]
data_of_versicolor_B=data_of_versicolor[25:, :]

data_of_verginica_A=data_of_verginica[:25, :]
data_of_verginica_B=data_of_verginica[25:, :]

X_A=np.vstack([data_of_verginica_A, data_of_versicolor_A])
X_B=np.vstack([data_of_verginica_B, data_of_versicolor_B])

Y_A = np.array([2] * len(data_of_verginica_A) + [1] * len(data_of_versicolor_A))
Y_B = np.array([2] * len(data_of_verginica_B) + [1] * len(data_of_versicolor_B))

#max_depth = [1, 3, 5, 7]
#
#fig, ax = plt.subplots(2, 4, sharex="col", sharey="row", figsize=(12, 6))
#
#
#j = 0
#for md in max_depth:
#    model = DecisionTreeClassifier(max_depth=md)
#    model.fit(X_A, Y_A)
#    Y_p = model.predict(X_p)
#    ax[0,j].scatter(data_of_verginica_A[:,0], data_of_verginica_A[:,1])
#    ax[0,j].scatter(data_of_versicolor_A[:,0], data_of_versicolor_A[:,1])
#    ax[0,j].contourf(X1_p, X2_p, Y_p.reshape(X1_p.shape), alpha=0.3, levels=[0.5, 1.5, 2.5])
#    j+=1
#
#j = 0
#for md in max_depth:
#    model = DecisionTreeClassifier(max_depth=md)
#    model.fit(X_B, Y_B)
#    Y_p = model.predict(X_p)
#    ax[1,j].scatter(data_of_verginica_B[:,0], data_of_verginica_B[:,1])
#    ax[1,j].scatter(data_of_versicolor_B[:,0], data_of_versicolor_B[:,1])
#    ax[1,j].contourf(X1_p, X2_p, Y_p.reshape(X1_p.shape), alpha=0.3, levels=[0.5, 1.5, 2.5])
#    j+=1


# Bagging
# Random Forest

#fig, ax = plt.subplots(1, 3, sharex="col", sharey="row", figsize=(12, 6))
#ax[0].scatter(data_of_setosa[:,0], data_of_setosa[:,1])
#ax[0].scatter(data_of_versicolor[:,0], data_of_versicolor[:,1])
#ax[0].scatter(data_of_verginica[:,0], data_of_verginica[:,1])
#
#ax[1].scatter(data_of_setosa[:,0], data_of_setosa[:,1])
#ax[1].scatter(data_of_versicolor[:,0], data_of_versicolor[:,1])
#ax[1].scatter(data_of_verginica[:,0], data_of_verginica[:,1])
#
#ax[2].scatter(data_of_setosa[:,0], data_of_setosa[:,1])
#ax[2].scatter(data_of_versicolor[:,0], data_of_versicolor[:,1])
#ax[2].scatter(data_of_verginica[:,0], data_of_verginica[:,1])
#
#model1 = DecisionTreeClassifier()
#model1.fit(data, Y)
#Y1_p=model1.predict(X_p)
#ax[0].contourf(X1_p, X2_p, Y1_p.reshape(X1_p.shape), alpha=0.3, levels=[-1, 0.5, 1.5, 2.5])
#
#from sklearn.ensemble import BaggingClassifier
#
#model2 = DecisionTreeClassifier()
#bagging=BaggingClassifier(model2, n_estimators=10, max_samples=0.6, random_state=1)
#bagging.fit(data, Y)
#
#Y2_p=bagging.predict(X_p)
#ax[1].contourf(X1_p, X2_p, Y2_p.reshape(X1_p.shape), alpha=0.3, levels=[-1, 0.5, 1.5, 2.5])
#
#from sklearn.ensemble import RandomForestClassifier
#
#model3=RandomForestClassifier(n_estimators=10, max_samples=0.6, random_state=1)
#model3.fit(data, Y)
#
#Y3_p=bagging.predict(X_p)
#ax[2].contourf(X1_p, X2_p, Y3_p.reshape(X1_p.shape), alpha=0.3, levels=[-1, 0.5, 1.5, 2.5])
#
#plt.show()

# Плюсы: простые модели, быстро решаются, параллелизм, голосование, непараметрическая
# Минусы: сложно сделать осмысленный вывод

#PCA

from sklearn.decomposition import PCA
X=iris.data
Y=iris.target

"""
X_iris [Длина чашелистника, Ширина чашелистника, Длина лепестка, Ширина лепестка]
Y_iris [0 - Ирис щетинистый, 1 - Ирис разноцветный, 2 - Ирис виргинский]
"""

data=X[Y==0][:,[0,2]]
target=Y[Y==0]

pca=PCA(n_components=2)
pca.fit(data)



fig=plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(pca.mean_[0], pca.mean_[1])
plt.plot(
    [pca.mean_[0], pca.mean_[0]+pca.components_[0][0]*np.sqrt(pca.explained_variance_[0])],
    [pca.mean_[1], pca.mean_[1]+pca.components_[0][1]*np.sqrt(pca.explained_variance_[0])],
)

plt.plot(
    [pca.mean_[0], pca.mean_[0]+pca.components_[1][0]*np.sqrt(pca.explained_variance_[1])],
    [pca.mean_[1], pca.mean_[1]+pca.components_[1][1]*np.sqrt(pca.explained_variance_[1])],
)

pca1=PCA(n_components=1)
pca1.fit(data)
X_pca1=pca1.transform(data)
X_new=pca1.inverse_transform(X_pca1)

plt.scatter(X_new[:,0], X_new[:,1])
plt.tight_layout()
plt.show()
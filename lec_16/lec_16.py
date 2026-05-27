"""
_____________________________________________
Линейная регрессия 

y=a0+a1*x1+a2*x2+a3*x3+...
_____________________________________________
Регрессия по комбинации базовых функций

y=a0+a1*x1+a2*x2+a3*x3+..., где xn=fn(x)
_____________________________________________
Логистическая регрессия 

Сигмойдная кривая y=1/(1+exp(-(m*x+b)))
_____________________________________________
Дерево решений 

                            x
                       |----|-----|
                       x          x
                  |----|-----|
                  x          x
_____________________________________________
Метод опорных векторов
"""

from sklearn.svm import SVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
x = iris.data
y = iris.target

x=x[y!=2][:, 0:2]
y=y[y!=2]

#print(x)
#print(y)

#model=SVC(kernel="linear", C=1e10)
#model.fit(x, y)
#
#xx, yy = np.meshgrid(
#    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
#    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
#)
#
#x_0=x[y==0][:,0]
#y_0=x[y==0][:,1]
#
#x_1=x[y==1][:,0]
#y_1=x[y==1][:,1]
#
#plt.scatter(x_0, y_0, color="red", alpha=0.5)
#plt.scatter(x_1, y_1, color="blue", alpha=0.5)
#
#Z=model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
#
#ax = plt.gca()
#
#ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5])
#
#plt.show()

"""
Наивная байесовская классификация
апостерионова вероятность P(A|B) = (P(B|A)*P(A))/P(B) 

L
P(L|признак) = (P(признак|L)*P(L))/P(признак)

L1 L2

P(L1|признак)/P(L2|признак) = (P(признак|L1)*P(L1))/(P(признак|L2)*P(L2))

"""

#from sklearn.naive_bayes import GaussianNB
#
#model=GaussianNB()
#
#model.fit(x, y)
#
#xx, yy = np.meshgrid(
#    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
#    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
#)
#
#x_0=x[y==0][:,0]
#y_0=x[y==0][:,1]
#
#x_1=x[y==1][:,0]
#y_1=x[y==1][:,1]
#
#plt.scatter(x_0, y_0, color="red", alpha=0.5)
#plt.scatter(x_1, y_1, color="blue", alpha=0.5)
#
#Z=model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
#
#ax = plt.gca()
#
#ax.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5])
#
#
#x_m=model.theta_[0]
#x_var=model.var_[0]
#y_m=model.theta_[1]
#y_var=model.var_[1]
#
#z1=1/(2*np.pi*(x_var[0]*x_var[1])**0.5)*np.exp(
#-((xx-x_m[0])**2) / (2 * x_var[0])
#- ((yy-x_m[1]) **2) / (2 * x_var[1])
#)
#ax.contourf(xx, yy, z1, alpha=0.3)
#
#z2=1/(2*np.pi*(y_var[0]*y_var[1])**0.5)*np.exp(
#-((xx-y_m[0])**2) / (2 * y_var[0])
#- ((yy-y_m[1]) **2) / (2 * y_var[1])
#)
#ax.contourf(xx, yy, z2, alpha=0.3)
#
#plt.show()
#
#ax=plt.axes(projection="3d")
#ax.contour3D(xx, yy, z1, 50)
#ax.contour3D(xx, yy, z2, 50)
#
#plt.show()

# k - ближайшие соседей

from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier()
model.fit(x, y)

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)

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



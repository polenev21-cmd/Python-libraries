import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
     

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

#x=np.linspace(-6, 6, 30)
#y=np.linspace(-10, 10, 50)
#
#X, Y=np.meshgrid(x, y)
#Z=f(X, Y)
#   
fig=plt.figure()
ax=plt.axes(projection='3d')
#
##ax.scatter3D(X, Y, Z, c=Z)
##ax.plot_wireframe(X, Y, Z)
#ax.plot_surface(X, Y, Z, cmap='viridis')
#
#angle=np.linspace(0, 2*np.pi, 50)
#r=np.linspace(0, 6, 30) 
#R, Angle=np.meshgrid(r, angle)
#
#ax=plt.axes(projection='3d')

#angle=np.linspace(0,  2*np.pi, 50)
#r=np.linspace(0, 6, 30) 
#
#R, Angle=np.meshgrid(r, angle)
#
#X=R*np.sin(Angle)
#Y=R*np.cos(Angle)
#Z=f(X, Y)
#
#ax.plot_surface(X, Y, Z, cmap='viridis')

#angle=1.5*np.pi*np.random.random(50).linspace(0,  1.5*np.pi, 50)
#r=np.linspace(0, 6, 1000) 
#
#R, Angle=np.meshgrid(r, angle)
#
#X=R*np.sin(Angle)
#Y=R*np.cos(Angle)
#Z=f(X, Y)
#
#x=r*np.sin(Angle)
#y=r*np.cos(Angle)
#z=f(x, y)
#
##ax.scatter3D(X, Y, Z, c=Z)
##
##ax.plot_surface(X, Y, Z, cmap='viridis')
#ax.plot_trisurf(x, y, z, cmap='viridis')


# Seaborn
import seaborn as sns
     
sns.set_style('darkgrid')
cars = pd.read_csv("cars.csv")

print(cars.head())

# Числовые данные
# Парная

#sns.pairplot(data=cars, hue="transmission")
#
## Тепловая карта
#
#cars_corr = cars[['year', "selling_price", "seats", "mileage"]]
#sns.heatmap(cars_corr.corr(), cmap="viridis", annot=True)


# Диаграмма рассеивания

#sns.scatterplot(x="seats", y='mileage', data=cars, hue="fuel")
#sns.scatterplot(x="year", y='selling_price', data=cars)

# Диаграмма рассеивания + линейная регрессия
#sns.relplot(x="seats", y='mileage', data=cars)
#sns.relplot(x="seats", y='mileage', data=cars, kind='scatter')
#sns.relplot(x="seats", y='mileage', data=cars, kind='scatter', hue='fuel')
#sns.relplot(x="seats", y='mileage', data=cars, kind='scatter', hue='fuel',
#            col_wrap=2, col='transmission')
#sns.relplot(x="seats", y='mileage', data=cars, kind='line', hue='fuel',
#            col_wrap=2, col='transmission')

# Диаграмма рассеивания + линейная регрессия

#sns.lmplot(data=cars, x="seats", y='mileage',
#            col='transmission', col_wrap=2, hue='fuel')

# Линейный график

#sns.lineplot(data=cars, x="seats", y='mileage')

# Сводные диаграммы

#sns.jointplot(x="year", y='selling_price', data=cars, kind='kde')
#
#sns.jointplot(x="year", y='selling_price', data=cars, kind='hex')
#
#sns.jointplot(x="year", y='selling_price', data=cars, hue='transmission')

# Категории и числа

#sns.barplot(x='fuel', y='selling_price', data=cars, estimator=np.mean)
#
#sns.barplot(x='fuel', y='selling_price', data=cars, estimator=np.mean, 
#            hue='transmission')
#
#sns.catplot(x='fuel', y='selling_price', data=cars, estimator=np.mean,
#            hue='transmission', col="seller_type", col_wrap=2)
#
#sns.pointplot(x='fuel', y='selling_price', data=cars,
#               estimator=np.mean, hue='transmission')


#sns.boxplot(x='fuel', y='selling_price', data=cars, hue='transmission')

sns.violinplot(x='fuel', y='selling_price', data=cars, hue='transmission', ax=ax)

sns.stripplot(x='fuel', y='selling_price', data=cars, hue='transmission', ax=ax, dodge=True)

plt.show()
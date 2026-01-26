import numpy as np
import sys
import array
print(np.__version__)

# динамическая типизация

x=1
print(type(x))
x="Hello world"
print(type(x))

l=[True, "2", 3.1, 4]
print (type(i) for i in l)
print(sys.getsizeof(l))

l1=[]
print (type(i) for i in l1)
print(sys.getsizeof(l1))

a1=array.array("i",[])
print(type(a1))
print(sys.getsizeof(a1))

a1=array.array("i",[1])
print(sys.getsizeof(a1))
a1=array.array("i",[1, 2])
print(sys.getsizeof(a1))

# Для больших массивов эффективнее однотипные списки. Такие массивы можно получить с помощью Numpy и array

l=[i for i in range(1000000)]
a=np.array(l)
print (type(a))

print("list(python)", sys.getsizeof(l))
ap=array.array("i", l)
print("array(python)", sys.getsizeof(ap))
a=np.array(l)
print("array(numpy)", sys.getsizeof(a))

# Повышающее приведение типов

a=np.array([1.1, 2, 4, 5, 6, 7])
print(type(a), a)

#Явно задать тип

a=np.array([1.1, 2, 4, 5, 6, 7], dtype=int)
print(type(a), a)

#Одномерный массив

a=np.array(range(2, 5))
print(a)

# Многомерный массив

a=np.array([range(i, i+5) for i in [1, 2, 3]])
print(a)

# с нуля

print(np.zeros(10, dtype=int))

print(np.zeros((3, 5), dtype=float))

# предопределённое значение

print(np.full((3,3), 3.1416))

# линейная последовательность чисел

print(np.arange(0, 20, 2))

# значения в интервале с одинаковыми промежутками 

print(np.linspace(0,  1, 11))

# равномерное распределение

print(np.random.random(2))

# нормальное распределениме

print(np.random.normal(0, 1, (2, 4)))

# равномерное распределение от x до у

print(np.random.randint(0, 5, (2, 2)))

# единичная марица

print(np.eye(5, dtype=int))

# Типы данных

al = np.zeros(10, dtype=int)
a2 = np.zeros (10, dtype='int16')
a3 = np.zeros(10, dtype=np.int16)
print(al, type (al), al.dtype) # python
print(a2, type (a2), a2.dtype) # np
print(a3, type (a3), a3.dtype) # np

# Numerical Python = NumPy
# - атрибуты массивов
# - индексация
# - срезы
# - изменение формы
# - объединение и разбиение

# Атрибуты: ndim - число размерностей, shape - размер каждой размерности, size - общий размер массива

np.random.seed(1)


x1=np.random.randint(10, size = 3)
print(x1)
print(x1.ndim, x1.shape, x1.size)

x2 = np.random.randint(10, size = (3, 2))
print(x2)
print(x2.ndim, x2.shape, x2.size)

# Индексация

a = np.array([1, 2, 3, 4, 5])
print(a[0])
print(a[-2])
a[1] = 20

print(a)

#многомерные

a=np.array([[1, 2],[3, 4]])
print(a)
print(a[0,0])

# вставки

a = np.array([1, 2, 3, 4, 5])
print(a.dtype)

a[0]=int("3")
print(a.dtype)

# типы на ходу менять не надо

# срезы - подмассив массива [начало:конец:шаг]

a = np.array([1, 2, 3, 4, 5])

print(a[:3])

print(a[3:])

print(a[1:4])

print(a[::2])

print(a[1::2])

# шаг меньше 0 - конец и начало меняются

a = np.array([1, 2, 3, 4, 5])
print(a[::-1])

# многомерные массивы

a = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])

print(a)

print(a[:2, :3])
print(a[::1, ::2])

print(a[::-1, ::-1])

print(a[:, 0])
print(a[0, :])
print(a[0])

# Срезы в питоне - копии массивов, в NumPy - представления массива

a = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])
print(a)

a_2x2=a[:2, :2]
print(a_2x2)

a_2x2[0, 0]=999
print(a)

# копия
a = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])
print(a)

a_2x2=a[:2, :2].copy()
print(a_2x2)

a_2x2[0, 0]=999
print(a_2x2)
print(a)

# форма массива

a=np.arange(1, 13)
print(a, a.shape, a.ndim)

print(a[3])
print(a[11])

a1=a.reshape(1, 12)
print(a1, a1.shape, a1.ndim)

print(a1[0, 3])
print(a1[0, 11])

a2=a.reshape(2, 6)
print(a2, a2.shape, a2.ndim)

a3=a.reshape(3, 4)
print(a3, a3.shape, a3.ndim)

a3=a.reshape(2, 2, 3)
print(a3, a3.shape, a3.ndim)
print([0, 1, 2])

a4=a.reshape(1, 12, 1, 1)
print(a4, a4.shape, a4.ndim)
print(a4[0, 2, 0, 0])

a5=a.reshape((2,6))
print(a5, a5.shape, a5.ndim)
print(a5[1, 5])

a6=a.reshape((2,6), order="F")
print(a6, a6.shape, a6.ndim)
print(a6[1, 4])


a=np.arange(1, 13)
print(a, a.shape, a.ndim)

a1=a.reshape(1, 12)
print(a1, a1.shape, a1.ndim)

a2=a[np.newaxis, :]
print(a2, a2.shape, a2.ndim)
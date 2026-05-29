import matplotlib.pyplot as plt
import numpy as np

"""Первый график"""

fig=plt.figure()

A=np.array([4, 3, 7, 0, 7, 12, 10, 5, 0, 0, 0])
B=np.array([1, 9, 6, 5, 8, 11, 14, 10, 7, 6, 1])
C=np.array([25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25])
O=np.arange(11)

plt.plot(O, A, linestyle="-.", color="green", marker="o", label="line 1")
plt.plot(O, B, color="red", marker="o", label="line 2")
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("1.jpg")

"""Второй график"""

fig=plt.figure()

ax1=plt.subplot(2, 2, (1, 2))
ax2=plt.subplot(2, 2, 3)
ax3=plt.subplot(2, 2, 4)

ax1.plot(O, A)
ax2.plot(O, B)
ax3.plot(O, C)

plt.tight_layout()
plt.savefig("2.jpg")

"""Третий график"""

fig=plt.figure()

plt.plot(O, C, color="blue")

plt.arrow(5, 10, 0, -10,
          width=0.15,
          head_length=2,
          length_includes_head=True,
          fc="green",
          ec="black",
          linewidth=2)

plt.text(5, 10, 
         "min", 
         fontsize=18, 
         color="black",
         ha="left",
         va="bottom")
plt.tight_layout()
plt.savefig("3.jpg")

"""Четвёртый график"""

fig=plt.figure()

x=np.linspace(0, 7, 7)
y=np.linspace(0, 7, 7)
X, Y=np.meshgrid(x, y)
Z=np.sin(X)+np.cos(Y)+np.sin(2*X)*np.cos(2*Y)

plt.imshow(Z, extent=[0, 7, 0, 7], cmap="viridis")
plt.colorbar(shrink=0.5, aspect=5, anchor=(0, 0))
plt.tight_layout()
plt.savefig("4.jpg")

"""Пятый график"""

fig=plt.figure()
x=np.linspace(0, 10, 100)
y=np.sin(x)

plt.plot(x, y, color="red", linestyle="-", linewidth=2)
plt.fill_between(x, y, 0, alpha=0.5, color="blue")
plt.tight_layout()
plt.savefig("5.jpg")

"""Шестой график"""

fig=plt.figure()

x=np.linspace(0, 10, 5000)
y=np.cos(np.pi * x)
y[y < -0.5]=np.nan

plt.figure(figsize=(8, 5))
plt.plot(x, y, linewidth=3)
plt.ylim(-1.0, 1.0)
plt.tight_layout()
plt.savefig("6.jpg")

"""Седьмой график"""

fig=plt.figure(figsize=(12, 4))

ax1=plt.subplot(1, 3, 1)
ax2=plt.subplot(1, 3, 2)
ax3=plt.subplot(1, 3, 3)

x=np.array([1, 2, 3, 4, 5, 6])
y=np.array([1, 2, 3, 4, 5, 6])

ax1.step(x, y, where='pre', color='green', marker="o")
ax1.grid(True)
ax2.step(x, y, where='post', color='green', marker="o")
ax2.grid(True)
ax3.step(x, y, where='mid', color='green', marker="o")
ax3.grid(True)

plt.tight_layout()
plt.savefig("7.jpg")

"""Восьмой график"""

plt.figure(figsize=(12, 6))

x=np.linspace(0, 10, 100)
y1=-x**2+12*x
y2=-0.6*x**2+6*x
y3=-0.2*x**2+2*x

plt.fill_between(x, y1, y2, label='y1', color='green')
plt.fill_between(x, y2, y3, label='y2', color='blue')
plt.fill_between(x, y3, 0, label='y3', color='red')

plt.plot(x, y1, linewidth=2, color='green')
plt.plot(x, y2, linewidth=2, color='blue')
plt.plot(x, y3, linewidth=2, color='red')

plt.legend(loc='upper left')
plt.ylim(0, max(y1)+5)
plt.tight_layout()
plt.savefig("8.jpg")

"""Девятый  график"""

plt.figure()
plt.pie([30, 25, 20, 15, 10], 
        labels=['Toyota', 'Ford', 'Jaguar', 'AUDI', 'BMW'], 
        explode=(0, 0.1, 0, 0, 0))
plt.tight_layout()
plt.savefig("9.jpg")

"""Десятый график"""

plt.figure()

plt.pie([30, 25, 20, 15, 10],
        labels=['Toyota', 'Ford', 'Jaguar', 'AUDI', 'BMW'],
        wedgeprops={'width': 0.5, 'edgecolor': 'white'})
plt.tight_layout()
plt.savefig("10.jpg")

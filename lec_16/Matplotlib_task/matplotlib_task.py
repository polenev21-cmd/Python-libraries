import matplotlib.pyplot as plt
import numpy as np

# Первый график
A=np.array([4, 3, 7, 0, 7, 12, 10, 5, 0, 0, 0])
B=np.array([1, 9, 6, 5, 8, 11, 14, 10, 7, 6, 1])
C=np.array([25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25])

O=np.arange(11)

plt.plot(O, A, linestyle="-.", color="green", marker="o", label="line 1")
plt.plot(O, B, color="red", marker="o", label="line 2")
plt.legend()
plt.savefig("First_plot.jpg")

# Второй график

fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(2, 2, (1, 2))
ax2 = plt.subplot(2, 2, 3)
ax3 = plt.subplot(2, 2, 4)

ax1.plot(O, A)
ax2.plot(O, B)
ax3.plot(O, C)

plt.tight_layout()
plt.savefig("Second_plot.jpg")

# Третий график
fig = plt.figure(figsize=(12, 8))

# Находим минимум
min_index = np.argmin(C)  # индекс минимума
min_value = np.min(C)      # значение минимума

plt.plot(O, C, 'ro-', linewidth=2, markersize=8)

# Стрелка к минимуму
plt.annotate('Минимум', 
             xy=(min_index, min_value),  # куда указываем
             xytext=(min_index + 1, min_value + 2),  # откуда рисуем
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.grid(True)
plt.savefig("Third_plot.jpg")

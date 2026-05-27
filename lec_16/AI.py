import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.linalg import svd

# ============================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================
iris = load_iris()
X = iris.data
y = iris.target

# Выбираем только два класса: Setosa (0) и Versicolor (1)
mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

print("Данные загружены:")
print(f"  Объектов: {X.shape[0]}")
print(f"  Признаков: {X.shape[1]}")
print(f"  Классы: Setosa (0) и Versicolor (1)")

# ============================================
# 2. МАСШТАБИРОВАНИЕ ДАННЫХ
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nДанные отмасштабированы (среднее=0, дисперсия=1)")

# ============================================
# 3. РЕАЛИЗАЦИЯ PCA ЧЕРЕЗ SVD (SciPy)
# ============================================
class PCACustom:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        # Центрирование
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # SVD через scipy
        U, s, Vt = svd(X_centered, full_matrices=False)
        
        # Главные компоненты
        self.components_ = Vt[:self.n_components]
        
        # Доля объясненной дисперсии
        variance = (s ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = variance[:self.n_components] / np.sum(variance)
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Применяем PCA
pca = PCACustom(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA выполнен:")
print(f"  Размерность снижена: {X.shape[1]} → {X_pca.shape[1]}")
print(f"  Сохранено дисперсии: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}% дисперсии")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}% дисперсии")

# ============================================
# 4. КЛАССИФИКАЦИЯ
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nКлассификация (Logistic Regression):")
print(f"  Точность на тесте: {accuracy*100:.1f}%")

# ============================================
# 5. ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================
fig = plt.figure(figsize=(16, 10))

# Цвета и метки
colors = ['red', 'blue']
labels = ['Setosa', 'Versicolor']

# ===== ГРАФИК 1: Исходные данные (первые два признака) =====
ax1 = fig.add_subplot(2, 3, 1)
for i in range(2):
    mask_class = (y == i)
    ax1.scatter(X[mask_class, 0], X[mask_class, 1], 
               c=colors[i], label=labels[i], s=60, alpha=0.7, edgecolors='black')
ax1.set_xlabel('Длина чашелистика (см)', fontsize=10)
ax1.set_ylabel('Ширина чашелистика (см)', fontsize=10)
ax1.set_title('Исходные данные\n(первые 2 признака)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ===== ГРАФИК 2: Исходные данные (последние два признака) =====
ax2 = fig.add_subplot(2, 3, 2)
for i in range(2):
    mask_class = (y == i)
    ax2.scatter(X[mask_class, 2], X[mask_class, 3], 
               c=colors[i], label=labels[i], s=60, alpha=0.7, edgecolors='black')
ax2.set_xlabel('Длина лепестка (см)', fontsize=10)
ax2.set_ylabel('Ширина лепестка (см)', fontsize=10)
ax2.set_title('Исходные данные\n(последние 2 признака)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ===== ГРАФИК 3: PCA - проекция =====
ax3 = fig.add_subplot(2, 3, 3)
for i in range(2):
    mask_class = (y == i)
    ax3.scatter(X_pca[mask_class, 0], X_pca[mask_class, 1], 
               c=colors[i], label=labels[i], s=80, alpha=0.7, edgecolors='black', linewidth=1.5)
ax3.set_xlabel(f'Первая главная компонента ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
ax3.set_ylabel(f'Вторая главная компонента ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
ax3.set_title('PCA - снижение размерности\n(классы в новом пространстве)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ===== ГРАФИК 4: Граница решения классификатора =====
ax4 = fig.add_subplot(2, 3, 4)
# Создаем сетку для визуализации границы
x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax4.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
for i in range(2):
    mask_class = (y == i)
    ax4.scatter(X_pca[mask_class, 0], X_pca[mask_class, 1], 
               c=colors[i], label=labels[i], s=60, alpha=0.7, edgecolors='black')
ax4.set_xlabel('Первая главная компонента', fontsize=10)
ax4.set_ylabel('Вторая главная компонента', fontsize=10)
ax4.set_title(f'Граница классификации в PCA-пространстве\n(точность = {accuracy*100:.1f}%)', 
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ===== ГРАФИК 5: Объясненная дисперсия =====
ax5 = fig.add_subplot(2, 3, 5)
components = [1, 2, 3, 4]
# Считаем для всех компонент
pca_full = PCACustom(n_components=4)
pca_full.fit(X_scaled)
ax5.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='steelblue', label='Индивидуальная')
ax5.plot(components, np.cumsum(pca_full.explained_variance_ratio_), 'ro-', linewidth=2, markersize=8, label='Накопленная')
ax5.set_xlabel('Номер главной компоненты', fontsize=10)
ax5.set_ylabel('Доля объясненной дисперсии', fontsize=10)
ax5.set_title('Объясненная дисперсия компонент PCA', fontsize=12, fontweight='bold')
ax5.set_xticks(components)
ax5.legend()
ax5.grid(True, alpha=0.3)
# Добавляем подписи значений
for i, v in enumerate(pca_full.explained_variance_ratio_):
    ax5.text(i+1, v + 0.02, f'{v*100:.1f}%', ha='center', fontsize=9)

# ===== ГРАФИК 6: Нагрузки признаков =====
ax6 = fig.add_subplot(2, 3, 6)
feature_names = ['Длина\nчашелистика', 'Ширина\nчашелистика', 'Длина\nлепестка', 'Ширина\nлепестка']
x_pos = np.arange(len(feature_names))
width = 0.35

ax6.bar(x_pos - width/2, pca.components_[0], width, label='PC1', alpha=0.7, color='coral')
ax6.bar(x_pos + width/2, pca.components_[1], width, label='PC2', alpha=0.7, color='lightgreen')
ax6.set_xlabel('Исходные признаки', fontsize=10)
ax6.set_ylabel('Вклад (нагрузка)', fontsize=10)
ax6.set_title('Вклад признаков в главные компоненты', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(feature_names, fontsize=8)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('КЛАССИФИКАЦИЯ ДВУХ СОРТОВ IRIS С ПРИМЕНЕНИЕМ PCA', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ============================================
# 6. ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: 3D визуализация PCA
# ============================================
fig = plt.figure(figsize=(12, 5))

# 3D график первых трех компонент
ax1 = fig.add_subplot(121, projection='3d')
pca_3d = PCACustom(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

for i in range(2):
    mask_class = (y == i)
    ax1.scatter(X_pca_3d[mask_class, 0], X_pca_3d[mask_class, 1], X_pca_3d[mask_class, 2],
               c=colors[i], label=labels[i], s=50, alpha=0.7, edgecolors='black')
ax1.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]*100:.1f}%)')
ax1.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]*100:.1f}%)')
ax1.set_title('3D визуализация PCA\n(первые 3 компоненты)', fontsize=12, fontweight='bold')
ax1.legend()

# Векторы главных компонент
ax2 = fig.add_subplot(122)
origin = np.zeros(4)
for i in range(4):
    ax2.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
             head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
    ax2.text(pca.components_[0, i], pca.components_[1, i], feature_names[i], fontsize=9)

# Добавляем точки данных в пространстве компонент
for i in range(2):
    mask_class = (y == i)
    ax2.scatter(X_pca[mask_class, 0], X_pca[mask_class, 1], 
               c=colors[i], label=labels[i], s=30, alpha=0.5)

ax2.set_xlim(-3, 4)
ax2.set_ylim(-3, 3)
ax2.set_xlabel('Первая главная компонента')
ax2.set_ylabel('Вторая главная компонента')
ax2.set_title('Проекция данных и векторы признаков\nв пространстве PCA', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 7. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ АНАЛИЗА")
print("="*60)
print(f"\n✅ Выбраны сорта: SETOSA и VERSICOLOR")
print(f"✅ Реализован PCA через SVD (scipy.linalg.svd)")
print(f"✅ Размерность снижена: 4 → 2 признака")
print(f"✅ Сохранено дисперсии: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
print(f"✅ Точность классификации: {accuracy*100:.1f}%")
print(f"\n📊 Интерпретация главных компонент:")
print(f"   PC1 (доля {pca.explained_variance_ratio_[0]*100:.1f}%): Ориентирована на признаки лепестка")
print(f"   PC2 (доля {pca.explained_variance_ratio_[1]*100:.1f}%): Ориентирована на признаки чашелистика")
print(f"\n💡 Вывод: Два сорта идеально разделимы в пространстве PCA")
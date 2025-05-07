import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Configurar el estilo de los gráficos
plt.style.use('seaborn-v0_8')
np.random.seed(42)

# 1. Crear datos de ejemplo de alta dimensionalidad (50D) con estructura de clusters
data_high_dim = np.random.rand(500, 50)  # 500 puntos en 50 dimensiones
# Añadimos estructura para que haya 5 clusters
for i in range(5):
    data_high_dim[i*100:(i+1)*100, i*10:(i+1)*10] += np.random.rand(100, 10) + 2

# Etiquetas de cluster para colorear
labels = np.repeat(np.arange(5), 100)

# Escalamos los datos
scaler = StandardScaler()
data_high_dim_scaled = scaler.fit_transform(data_high_dim)

# 2. Comparamos PCA y t-SNE
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

data_pca = pca.fit_transform(data_high_dim_scaled)
data_tsne = tsne.fit_transform(data_high_dim_scaled)

# Visualizar la comparativa
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter1 = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
plt.title('PCA (2 componentes)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
cbar1 = plt.colorbar(scatter1)
cbar1.set_label('Cluster')

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
plt.title('t-SNE (2 componentes)')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
cbar2 = plt.colorbar(scatter2)
cbar2.set_label('Cluster')

plt.tight_layout()
plt.savefig('pca_vs_tsne.png', dpi=300, bbox_inches='tight')

# 3. Aplicar PCA y t-SNE a datos con formas complejas
datasets = [
    ("Blobs", *make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)),
    ("Moons", *make_moons(n_samples=200, noise=0.05, random_state=42)),
    ("Circles", *make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42))
]

plt.figure(figsize=(15, 10))

for i, (name, X, y) in enumerate(datasets):
    # Datos originales (ya están en 2D para visualización)
    plt.subplot(3, 3, i*3+1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
    plt.title(f'Original: {name}')
    
    # Aumentar artificialmente la dimensionalidad
    high_dim = np.hstack([X, np.random.randn(X.shape[0], 10) * 0.1])
    
    # Aplicar PCA
    X_pca = PCA(n_components=2).fit_transform(high_dim)
    plt.subplot(3, 3, i*3+2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
    plt.title(f'PCA: {name}')
    
    # Aplicar t-SNE
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(high_dim)
    plt.subplot(3, 3, i*3+3)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=30, alpha=0.7)
    plt.title(f't-SNE: {name}')

plt.tight_layout()
plt.savefig('formas_complejas.png', dpi=300, bbox_inches='tight')

# 4. Análisis de varianza explicada en PCA
pca_full = PCA().fit(data_high_dim_scaled)
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(12, 5))

# Gráfico de codo para varianza explicada
plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2)
plt.title('Varianza explicada por componente')
plt.xlabel('Número de componente')
plt.ylabel('Proporción de varianza explicada')
plt.grid(True)

# Gráfico de varianza acumulada
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', linewidth=2)
plt.axhline(y=0.9, color='r', linestyle='--', label='90% de varianza')
plt.title('Varianza acumulada explicada')
plt.xlabel('Número de componentes')
plt.ylabel('Proporción acumulada')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('varianza_pca.png', dpi=300, bbox_inches='tight')

# 5. Efecto de la perplexidad en t-SNE
perplexities = [5, 30, 50, 100]
plt.figure(figsize=(15, 10))

for i, perp in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    data_tsne = tsne.fit_transform(data_high_dim_scaled)
    
    plt.subplot(2, 2, i+1)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
    plt.title(f't-SNE (perplexidad={perp})')
    plt.colorbar(label='Cluster')

plt.tight_layout()
plt.savefig('perplexidad_tsne.png', dpi=300, bbox_inches='tight')

# 6. PCA a 3D como paso intermedio para visualización
pca_3d = PCA(n_components=3).fit_transform(data_high_dim_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], 
                     c=labels, cmap='viridis', s=30, alpha=0.7)
ax.set_title('PCA - Reducción a 3D')
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
ax.set_zlabel('Componente 3')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('pca_3d.png', dpi=300, bbox_inches='tight')

# 7. Cadena de reducción: PCA seguido de t-SNE
# Primero reducimos a 10 dimensiones con PCA
pca_10d = PCA(n_components=10).fit_transform(data_high_dim_scaled)

# Luego aplicamos t-SNE a esas 10 dimensiones
tsne_after_pca = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(pca_10d)

plt.figure(figsize=(10, 8))
plt.scatter(tsne_after_pca[:, 0], tsne_after_pca[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
plt.title('PCA (50D → 10D) + t-SNE (10D → 2D)')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig('pca_tsne_cadena.png', dpi=300, bbox_inches='tight')

# Mostrar todas las figuras
plt.show()

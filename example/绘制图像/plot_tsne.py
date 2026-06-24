##########  python plot_tsne.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 1. 加载特征数据
data = np.load('../tda_features.npz')
X = data['X']
y = data['y']

# 2. 分离恒星和星系
star_mask = (y == 0)
galaxy_mask = (y == 1)
X_stars = X[star_mask]
X_galaxies = X[galaxy_mask]

print(f"Total stars: {len(X_stars)}")
print(f"Total galaxies: {len(X_galaxies)}")

# 3. 减少星系采样数（关键加速点）
n_galaxy_sample = 3000
np.random.seed(42)
idx_gal = np.random.choice(len(X_galaxies), n_galaxy_sample, replace=False)
X_galaxies_sampled = X_galaxies[idx_gal]
y_galaxies_sampled = y[galaxy_mask][idx_gal]

# 4. 合并
X_sample = np.vstack([X_stars, X_galaxies_sampled])
y_sample = np.concatenate([np.zeros(len(X_stars)), np.ones(n_galaxy_sample)])

# 5. 打乱
np.random.seed(426)
shuffle_idx = np.random.permutation(len(X_sample))
X_sample = X_sample[shuffle_idx]
y_sample = y_sample[shuffle_idx]

print(f"Final samples: {len(X_sample)} (stars {len(X_stars)} + galaxies {n_galaxy_sample})")

# 6. PCA 降维到 50 维
print("Applying PCA (656 -> 50)...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_sample)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# 7. t-SNE
print("Running t-SNE (50D -> 2D)")
tsne = TSNE(n_components=2, random_state=42, perplexity=20, n_iter=500, n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca)

# 8. 绘图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='coolwarm', s=10, alpha=0.7)
plt.legend(handles=scatter.legend_elements()[0], labels=['Stars', 'Galaxies'])
plt.title('TDA Feature Distribution(t-SNE)')
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.savefig('tsne_optimized.png', dpi=150)
plt.close()
print("Image saved as tsne_optimized.png")
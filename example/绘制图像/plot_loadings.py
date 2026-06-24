##########  python plot_loadings.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 1. 加载数据
data = np.load('../tda_features.npz')
X = data['X']
y = data['y']

# 2. 标准化（重要！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 拟合PCA
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)
components = pca.components_  # 载荷矩阵 (50, 656)

pc1_weights = components[0]
pc2_weights = components[1]

# ===== 绘制带特征类型分区的载荷图 =====
fig, ax = plt.subplots(figsize=(16, 7))

# ---- 子区域定义 ----
sub_regions = [
    (0, 39, 'Betti', '#D8BFD8'),      # 紫色
    (40, 41, 'Entropy', '#F5DEB3'),   # 小麦色
    (42, 161, 'Landscape', '#B0E0E6'),# 粉蓝
    (162, 163, 'Amplitude', '#FADADD'),# 粉色
]

# ---- 每个图像块的范围 ----
block_ranges = [(0, 163), (164, 327), (328, 491), (492, 655)]

# ---- 在每个块内绘制子区域色块（无边框） ----
for b_start, b_end in block_ranges:
    for s_start, s_end, _, color in sub_regions:
        x1 = b_start + s_start
        x2 = b_start + s_end
        ax.axvspan(x1, x2, alpha=0.25, color=color, edgecolor='none')

# ---- 绘制载荷曲线 ----
ax.plot(pc1_weights, alpha=0.8, color='blue', linewidth=1.5, label='PC1 Loadings')
ax.plot(pc2_weights, alpha=0.8, color='orange', linewidth=1.5, label='PC2 Loadings')

# ---- 手动构建图例 ----
legend_elements = [
    Line2D([0], [0], color='blue', lw=1.5, label='PC1 Loadings'),
    Line2D([0], [0], color='orange', lw=1.5, label='PC2 Loadings'),
    Patch(facecolor='#D8BFD8', label='Betti Curve'),
    Patch(facecolor='#F5DEB3', label='Entropy'),
    Patch(facecolor='#B0E0E6', label='Landscape'),
    Patch(facecolor='#FADADD', label='Amplitude')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# ---- 装饰 ----
ax.set_xlabel('Original Feature Index (0-655)')
ax.set_ylabel('Loading Weight')
ax.set_title('PCA Loadings Distribution)
ax.grid(alpha=0.3)
ax.set_xlim(0, 655)

plt.tight_layout()
plt.savefig('pca_loadings.png', dpi=150)
plt.close()

print("标准化后的载荷分布图已保存为 pca_loadings.png")
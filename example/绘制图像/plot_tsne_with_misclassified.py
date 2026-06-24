##########  python plot_tsne_with_misclassified.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

# 路径规范化函数（处理不同路径格式）
def normalize_path(p):
    """去掉开头的 ./ 和 ../，统一路径格式"""
    p = p.lstrip('./').lstrip('../')
    return p

# 1. 加载特征数据
data = np.load('../tda_features.npz')
X = data['X']
y = data['y']
paths = data['paths']   # 所有样本路径（与特征顺序一致）

# 2. 读取分类错误样本路径
mis_paths_file = '../misclassified_paths.txt'
if not os.path.exists(mis_paths_file):
    raise FileNotFoundError(f"错误：找不到 {mis_paths_file}，请先运行训练脚本生成该文件。")

with open(mis_paths_file, 'r') as f:
    mis_paths = [line.strip() for line in f if line.strip()]

print(f"错误样本数: {len(mis_paths)}")

# 3. 构建路径→索引映射（规范化后匹配）
path_to_idx = {normalize_path(p): i for i, p in enumerate(paths)}
mis_indices = []
for p in mis_paths:
    norm = normalize_path(p)
    if norm in path_to_idx:
        mis_indices.append(path_to_idx[norm])
    else:
        print(f"Warning: 路径 {p} 在特征文件中未找到")

# 4. 分离错误样本
mis_indices = np.array(mis_indices)
if len(mis_indices) == 0:
    print("没有错误样本，直接绘制普通 t-SNE 图")
    # 可以回到原始代码，或直接退出
    exit()

# 5. 构建绘图数据集：包含所有恒星 + 所有错误星系 + 随机抽样正确星系
# 获取所有恒星的索引（包括正确和错误的）
star_indices = np.where(y == 0)[0]
# 获取错误恒星和错误星系的索引
mis_star_indices = mis_indices[y[mis_indices] == 0]
mis_gal_indices = mis_indices[y[mis_indices] == 1]

# 正确星系索引（排除错误星系）
correct_gal_indices = np.where((y == 1) & (~np.isin(np.arange(len(y)), mis_indices)))[0]
# 随机抽样正确星系（数量为 3000 - len(mis_gal_indices)）
n_gal_sample = max(0, 3000 - len(mis_gal_indices))
if n_gal_sample > 0:
    np.random.seed(42)
    sampled_gal_indices = np.random.choice(correct_gal_indices, 
                                          size=min(n_gal_sample, len(correct_gal_indices)), 
                                          replace=False)
else:
    sampled_gal_indices = np.array([])

# 合并索引
selected_indices = np.concatenate([star_indices, mis_gal_indices, sampled_gal_indices])
X_selected = X[selected_indices]
y_selected = y[selected_indices]

# 记录哪些是错误样本（用于标记）
# 错误样本在 selected_indices 中的位置
mis_selected_mask = np.isin(selected_indices, mis_indices)

print(f"绘图样本总数: {len(X_selected)} (恒星: {len(star_indices)}, 错误星系: {len(mis_gal_indices)}, 正确星系抽样: {len(sampled_gal_indices)})")
print(f"其中错误样本: {np.sum(mis_selected_mask)}")

# 6. PCA 降维
print("Applying PCA (656 -> 50)...")
pca = PCA(n_components=150, random_state=42)
X_pca = pca.fit_transform(X_selected)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# 7. t-SNE
print("Running t-SNE (50D -> 2D)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500, n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca)

# 8. 绘图
plt.figure(figsize=(10, 8))

# 正常样本（非错误）用颜色表示类别
normal_mask = ~mis_selected_mask
plt.scatter(X_tsne[normal_mask, 0], X_tsne[normal_mask, 1], 
            c=y_selected[normal_mask], cmap='coolwarm', s=10, alpha=0.7)

# 错误样本用黑色空心圆圈高亮
plt.scatter(X_tsne[mis_selected_mask, 0], X_tsne[mis_selected_mask, 1], 
            facecolors='none', edgecolors='black', s=100, linewidths=2, label='Misclassified')

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Stars',
           markerfacecolor='red', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Galaxies',
           markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Misclassified',
           markerfacecolor='none', markeredgecolor='black', markersize=10)
]
plt.legend(handles=legend_elements, loc='best')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('tsne_misclassified.png', dpi=150)
plt.close()
print("图像已保存为 tsne_misclassified.png")
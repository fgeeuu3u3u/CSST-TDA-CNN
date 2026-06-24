##########  python plot_betti_misclassified.py

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 配置 ==========
BETTI_BINS = 20
FEATURE_BLOCK = 'GI_inv'
BLOCK_OFFSETS = {
    'GI_orig': 0,
    'GV_orig': 164,
    'GI_inv': 328,
    'GV_inv': 492,
}
offset = BLOCK_OFFSETS[FEATURE_BLOCK]
h0_cols = np.arange(offset, offset + BETTI_BINS)
h1_cols = np.arange(offset + BETTI_BINS, offset + 2 * BETTI_BINS)

# ========== 1. 加载数据 ==========
data = np.load('../tda_features.npz', allow_pickle=True)
X = data['X']
y = data['y']
paths = data['paths']

with open('../test_paths.txt', 'r') as f:
    test_paths = [line.strip() for line in f if line.strip()]
with open('../misclassified_paths.txt', 'r') as f:
    mis_paths = [line.strip() for line in f if line.strip()]

test_set = set(test_paths)
mis_set = set(mis_paths)
path_to_idx = {p: i for i, p in enumerate(paths)}

test_indices = [path_to_idx[p] for p in test_paths if p in path_to_idx]
mis_indices = [path_to_idx[p] for p in mis_paths if p in path_to_idx]

star_indices = [i for i in test_indices if y[i] == 0]
gal_indices = [i for i in test_indices if y[i] == 1]

correct_star_idx = [i for i in star_indices if i not in mis_indices]
mis_star_idx = [i for i in mis_indices if y[i] == 0]
correct_gal_idx = [i for i in gal_indices if i not in mis_indices]

print(f"正确恒星: {len(correct_star_idx)}, 误分类恒星: {len(mis_star_idx)}, 星系: {len(correct_gal_idx)}")

# ========== 2. 提取 Betti 曲线 ==========
def get_betti_curves(indices, cols):
    if len(indices) == 0:
        return np.empty((0, BETTI_BINS))
    return X[indices][:, cols]

h0_correct = get_betti_curves(correct_star_idx, h0_cols)
h0_mis = get_betti_curves(mis_star_idx, h0_cols)
h0_gal = get_betti_curves(correct_gal_idx, h0_cols)

h1_correct = get_betti_curves(correct_star_idx, h1_cols)
h1_mis = get_betti_curves(mis_star_idx, h1_cols)
h1_gal = get_betti_curves(correct_gal_idx, h1_cols)

# ========== 3. 绘图 ==========
thresholds = np.linspace(0, 1, BETTI_BINS)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_curve(ax, curves, color, label):
    if len(curves) == 0:
        return
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    line, = ax.plot(thresholds, mean, color=color, linewidth=2, label=label)
    ax.fill_between(thresholds, mean - std, mean + std, color=color, alpha=0.2)
    return line

# H0 子图（显示图例）
plot_curve(ax1, h0_correct, 'red', f'Correct Stars (n={len(h0_correct)})')
plot_curve(ax1, h0_mis, 'orange', f'Misclassified Stars (n={len(h0_mis)})')
plot_curve(ax1, h0_gal, 'blue', f'Galaxies (n={len(h0_gal)})')
ax1.set_title('H0 Betti Curves')
ax1.set_xlabel('Filtration Threshold')
ax1.set_ylabel('H0 Betti Number')
ax1.legend(loc='best')   # 左图显示图例

# H1 子图（不显示图例）
plot_curve(ax2, h1_correct, 'red', f'Correct Stars (n={len(h1_correct)})')
plot_curve(ax2, h1_mis, 'orange', f'Misclassified Stars (n={len(h1_mis)})')
plot_curve(ax2, h1_gal, 'blue', f'Galaxies (n={len(h1_gal)})')
ax2.set_title('H1 Betti Curves')
ax2.set_xlabel('Filtration Threshold')
ax2.set_ylabel('H1 Betti Number')
# 此处不调用 ax2.legend()

plt.tight_layout()
plt.savefig('betti_all_samples.png', dpi=150)
plt.close()
print("图像已保存为 betti_all_samples.png")
##########  python plot_results.py

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ========== 配置 ==========
FEATURES_FILE = '../tda_features.npz'
SNR_CACHE = 'snr_values.npz'
TEST_PATHS_FILE = '../test_paths.txt'
MODEL_FILE = '../star_galaxy_classifier.joblib'

BINS = [0, 1, 3, 5, 10, 20, 50, 100]
BIN_LABELS = ['<1', '1-3', '3-5', '5-10', '10-20', '20-50', '50-100', '>100']

# ========== 1. 加载 SNR 字典 ==========
def load_snr_dict():
    data = np.load(SNR_CACHE, allow_pickle=True)
    snr_dict = {}
    for p, s in zip(data['paths_galaxies'], data['snr_galaxies']):
        snr_dict[p] = s
    for p, s in zip(data['paths_stars'], data['snr_stars']):
        snr_dict[p] = s
    return snr_dict

# ========== 2. 加载测试集索引 ==========
def get_test_indices():
    data = np.load(FEATURES_FILE, allow_pickle=True)
    paths_all = data['paths']
    with open(TEST_PATHS_FILE, 'r') as f:
        test_paths = [line.strip() for line in f if line.strip()]
    path_to_idx = {p: i for i, p in enumerate(paths_all)}
    indices = []
    for p in test_paths:
        if p in path_to_idx:
            indices.append(path_to_idx[p])
        else:
            clean = p.lstrip('./').lstrip('../')
            if clean in path_to_idx:
                indices.append(path_to_idx[clean])
            else:
                print(f"Warning: {p} not found")
    return np.array(indices)

# ========== 3. 主评估 ==========
def main():
    # 检查文件
    for f in [FEATURES_FILE, SNR_CACHE, TEST_PATHS_FILE, MODEL_FILE]:
        if not os.path.exists(f):
            print(f"Error: {f} not found")
            return

    # 加载特征和标签
    data = np.load(FEATURES_FILE, allow_pickle=True)
    X_all, y_all, paths_all = data['X'], data['y'], data['paths']

    # 获取测试集索引
    test_indices = get_test_indices()
    print(f"Test set size: {len(test_indices)}")

    # 取出测试集
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    paths_test = paths_all[test_indices]

    # 加载模型并预测
    model_data = joblib.load(MODEL_FILE)
    pipeline = model_data.get('pipeline', model_data.get('classifier'))
    y_pred = pipeline.predict(X_test)

    # 加载 SNR 字典
    snr_dict = load_snr_dict()

    # 为测试集获取 SNR
    snr_test = []
    for p in paths_test:
        if p in snr_dict:
            snr_test.append(snr_dict[p])
        else:
            clean = p.lstrip('./').lstrip('../')
            if clean in snr_dict:
                snr_test.append(snr_dict[clean])
            else:
                snr_test.append(np.nan)
    snr_test = np.array(snr_test)

    # 剔除 SNR 无效的样本
    valid = ~np.isnan(snr_test)
    if np.sum(~valid) > 0:
        print(f"Warning: {np.sum(~valid)} samples have no SNR, excluded.")
    snr_test = snr_test[valid]
    y_test = y_test[valid]
    y_pred = y_pred[valid]

    # 整体恒星召回率（对照）
    star_mask_all = (y_test == 0)
    if np.sum(star_mask_all) > 0:
        overall_recall = np.mean(y_test[star_mask_all] == y_pred[star_mask_all])
        print(f"Overall star recall (valid samples): {overall_recall:.4f}")
    else:
        print("No stars in valid samples!")
        return

    # 分箱
    bin_indices = np.digitize(snr_test, BINS, right=False)

    results = []
    for i in range(1, len(BINS)):
        mask = (bin_indices == i)
        if not np.any(mask):
            continue
        n_total = np.sum(mask)
        star_mask = mask & (y_test == 0)
        n_star = np.sum(star_mask)
        gal_mask = mask & (y_test == 1)
        n_gal = np.sum(gal_mask)

        if n_star > 0:
            correct = np.sum(y_test[star_mask] == y_pred[star_mask])
            star_rec = correct / n_star
            # 精确率：预测为恒星且正确的 / 预测为恒星的总数
            pred_star_mask = mask & (y_pred == 0)
            n_pred_star = np.sum(pred_star_mask)
            if n_pred_star > 0:
                star_prec = np.sum((y_test[pred_star_mask] == 0) & (y_pred[pred_star_mask] == 0)) / n_pred_star
            else:
                star_prec = np.nan
        else:
            star_rec, star_prec = np.nan, np.nan

        if n_gal > 0:
            gal_correct = np.sum(y_test[gal_mask] == y_pred[gal_mask])
            gal_rec = gal_correct / n_gal
            # 星系精确率
            pred_gal_mask = mask & (y_pred == 1)
            n_pred_gal = np.sum(pred_gal_mask)
            if n_pred_gal > 0:
                gal_prec = np.sum((y_test[pred_gal_mask] == 1) & (y_pred[pred_gal_mask] == 1)) / n_pred_gal
            else:
                gal_prec = np.nan
        else:
            gal_rec, gal_prec = np.nan, np.nan

        results.append({
            'label': BIN_LABELS[i-1],
            'n_total': n_total,
            'n_star': n_star,
            'n_gal': n_gal,
            'star_recall': star_rec,
            'star_precision': star_prec,
            'gal_recall': gal_rec,
            'gal_precision': gal_prec
        })
        print(f"Bin {BIN_LABELS[i-1]}: stars={n_star}, star recall={star_rec:.3f}")

    # 输出表格
    with open('snr_performance_both.txt', 'w') as f:
        f.write("SNR bin\tTotal\tStars\tStar Recall\tStar Precision\tGalaxies\tGalaxy Recall\tGalaxy Precision\n")
        for r in results:
            s_rec = f"{r['star_recall']:.3f}" if not np.isnan(r['star_recall']) else "N/A"
            s_prec = f"{r['star_precision']:.3f}" if not np.isnan(r['star_precision']) else "N/A"
            g_rec = f"{r['gal_recall']:.3f}" if not np.isnan(r['gal_recall']) else "N/A"
            g_prec = f"{r['gal_precision']:.3f}" if not np.isnan(r['gal_precision']) else "N/A"
            f.write(f"{r['label']}\t{r['n_total']}\t{r['n_star']}\t{s_rec}\t{s_prec}\t{r['n_gal']}\t{g_rec}\t{g_prec}\n")
    print("Table saved to snr_performance_both.txt")

    # 绘图
    labels = [r['label'] for r in results]
    star_rec = [r['star_recall'] for r in results]
    star_prec = [r['star_precision'] for r in results]
    gal_rec = [r['gal_recall'] for r in results]
    gal_prec = [r['gal_precision'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(labels, star_rec, 'o-', label='Star Recall', linewidth=2, markersize=8, color='red')
    plt.plot(labels, star_prec, 's--', label='Star Precision', linewidth=2, markersize=8, color='darkred')
    plt.plot(labels, gal_rec, '^-', label='Galaxy Recall', linewidth=2, markersize=8, color='blue')
    plt.plot(labels, gal_prec, 'd--', label='Galaxy Precision', linewidth=2, markersize=8, color='darkblue')
    plt.axhline(y=0.85, color='gray', linestyle='--', label='Reference (0.85)')
    plt.xlabel('Signal-to-Noise Ratio (SNR)')
    plt.ylabel('Score')
    plt.ylim(0.45, 1.05)
    plt.legend(loc='lower right')
    plt.title('Classification Performance')
    plt.tight_layout()
    plt.savefig('snr_performance_both.png', dpi=150)
    plt.close()
    print("Plot saved to snr_performance_both.png")

    # 最低SNR
    for r in results:
        if r['n_star'] > 0 and not np.isnan(r['star_recall']) and r['star_recall'] < 0.85:
            print(f"Recommended minimum SNR: {r['label']} (recall {r['star_recall']:.3f})")
            return
    print("All bins with stars have recall >= 0.85. Minimum SNR can be set to ≥1.")

if __name__ == "__main__":
    main()
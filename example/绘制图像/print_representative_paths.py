##########  python print_representative_paths.py

import numpy as np
import os

# ========== 配置 ==========
SNR_CACHE = 'snr_values.npz'
FEATURES_FILE = '../tda_features.npz'   # 用于获取所有路径和标签
BINS = [0, 1, 3, 5, 10, 20, 50, 100]
BIN_LABELS = ['<1', '1-3', '3-5', '5-10', '10-20', '20-50', '50-100', '>100']

def load_snr_dict():
    """从SNR缓存构建路径->SNR字典"""
    data = np.load(SNR_CACHE, allow_pickle=True)
    snr_dict = {}
    for p, s in zip(data['paths_galaxies'], data['snr_galaxies']):
        snr_dict[p] = s
    for p, s in zip(data['paths_stars'], data['snr_stars']):
        snr_dict[p] = s
    return snr_dict

def main():
    # 1. 加载所有特征路径和标签
    data = np.load(FEATURES_FILE, allow_pickle=True)
    paths_all = data['paths']   # 所有样本路径
    labels_all = data['y']      # 所有样本标签（0=恒星，1=星系）
    print(f"Total samples in features: {len(paths_all)}")

    # 2. 加载SNR字典
    snr_dict = load_snr_dict()
    print(f"SNR entries loaded: {len(snr_dict)}")

    # 3. 为所有路径匹配SNR（若匹配不到则跳过）
    snr_list = []
    valid_paths = []
    valid_labels = []
    for p, lab in zip(paths_all, labels_all):
        if p in snr_dict:
            snr_list.append(snr_dict[p])
            valid_paths.append(p)
            valid_labels.append(lab)
        else:
            # 尝试清理路径
            clean = p.lstrip('./').lstrip('../')
            if clean in snr_dict:
                snr_list.append(snr_dict[clean])
                valid_paths.append(p)
                valid_labels.append(lab)
            else:
                # 可选：打印警告，但可能数量很多，暂时忽略
                pass

    if not valid_paths:
        print("No valid paths with SNR found.")
        return

    snr_arr = np.array(snr_list)
    paths_arr = np.array(valid_paths)
    labels_arr = np.array(valid_labels)

    print(f"Valid samples with SNR: {len(snr_arr)}")

    # 4. 分箱
    bin_indices = np.digitize(snr_arr, BINS, right=False)

    # 5. 对每个箱体选一个恒星和一个星系
    print("="*60)
    print("Representative paths for each SNR bin (all samples):")
    print("")

    results = []
    for i in range(1, len(BINS)):
        mask = (bin_indices == i)
        if not np.any(mask):
            continue
        star_mask = mask & (labels_arr == 0)
        gal_mask = mask & (labels_arr == 1)

        star_path = ""
        gal_path = ""

        if np.any(star_mask):
            idx = np.where(star_mask)[0][0]  # 取第一个
            star_path = paths_arr[idx]
        if np.any(gal_mask):
            idx = np.where(gal_mask)[0][0]
            gal_path = paths_arr[idx]

        print(f"SNR bin {BIN_LABELS[i-1]}:")
        print(f"  Star:  {star_path if star_path else '(none)'}")
        print(f"  Galaxy: {gal_path if gal_path else '(none)'}")
        print("")
        results.append((BIN_LABELS[i-1], star_path, gal_path))

    # 6. 保存到文件
    with open('representative_paths_all.txt', 'w') as f:
        f.write("SNR bin\tStar path\tGalaxy path\n")
        for label, sp, gp in results:
            f.write(f"{label}\t{sp}\t{gp}\n")
    print("Representative paths saved to representative_paths_all.txt")

if __name__ == "__main__":
    main()
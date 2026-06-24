#!/usr/bin/env python3
"""
===============================================================================
星系-恒星拓扑特征提取脚本

功能：
    从GI和GV波段的二维光谱FITS图像中提取拓扑特征（Betti曲线、持久熵、持续性景观、振幅），
    将特征矩阵、标签、样本路径保存为 .npz 文件，供后续训练使用。

使用方法：
    python extract_features.py [选项]

核心参数：
    --cores N                   CPU核心数（默认全部）
    --use-original / --no-original        使用/禁用正图像（吸收线敏感，默认启用）
    --use-inverted / --no-inverted        使用/禁用反图像（发射线敏感，默认启用）
    --features [betti entropy landscape amplitude]   选择特征（默认betti entropy）
    --betti-bins N              贝蒂曲线离散点数（默认20）
    --landscape-layers N        持续性景观层数（默认2）
    --landscape-bins N          持续性景观离散点数（默认50）
    --amplitude-metric {bottleneck,wasserstein,heat}  振幅度量（默认bottleneck）

示例：
    # 默认（正反图像 + 贝蒂曲线+持久熵）
    python extract_features.py

    # 全特征高分辨率
    python extract_features.py --features betti entropy landscape amplitude --betti-bins 20 --landscape-layers 3 --landscape-bins 20 --amplitude-metric wasserstein --cores 26

注意事项：
    - 特征维度：贝蒂曲线维度 = betti_bins × 2；正反图像同时使用则特征维度翻倍。
    - 输出文件：tda_features.npz（包含 X, y, paths）
===============================================================================
"""

import numpy as np
import glob
import os
import argparse
from astropy.io import fits
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, BettiCurve, PersistenceEntropy, PersistenceLandscape, Amplitude
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# ---------- 特征提取函数----------
def extract_features(fits_path, invert, feature_config):
    """特征提取，返回特征向量"""
    img = fits.getdata(fits_path)
    img_norm = (img - img.min()) / (img.max() - img.min())
    if invert:
        img_norm = 1.0 - img_norm
    X = img_norm[np.newaxis, :, :]

    cubical = CubicalPersistence(homology_dimensions=(0, 1), n_jobs=1)
    diagrams = cubical.fit_transform(X)
    scaler = Scaler()
    diag_scaled = scaler.fit_transform(diagrams)

    feature_list = []

    if feature_config.get('betti', False):
        n_bins = feature_config.get('betti_bins', 20)
        betti = BettiCurve(n_bins=n_bins, n_jobs=1)
        betti_curves = betti.fit_transform(diag_scaled)
        betti_h0 = betti_curves[0, 0, :]
        betti_h1 = betti_curves[0, 1, :]
        feature_list.append(betti_h0)
        feature_list.append(betti_h1)

    if feature_config.get('entropy', False):
        entropy = PersistenceEntropy(nan_fill_value=-1, n_jobs=1)
        entropy_vals = entropy.fit_transform(diag_scaled)
        ent_h0 = entropy_vals[0, 0]
        ent_h1 = entropy_vals[0, 1]
        feature_list.append(np.array([ent_h0, ent_h1]))

    if feature_config.get('landscape', False):
        n_layers = feature_config.get('landscape_n_layers', 2)
        n_bins = feature_config.get('landscape_n_bins', 50)
        landscape = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins, n_jobs=1)
        landscape_vals = landscape.fit_transform(diag_scaled)[0]
        feature_list.append(landscape_vals.flatten())

    if feature_config.get('amplitude', False):
        metric = feature_config.get('amplitude_metric', 'bottleneck')
        metric_params = feature_config.get('amplitude_params', {})
        amplitude = Amplitude(metric=metric, metric_params=metric_params, n_jobs=1)
        amp_vals = amplitude.fit_transform(diag_scaled)[0]
        feature_list.append(amp_vals)

    if not feature_list:
        raise ValueError("至少需要选择一种拓扑特征")

    return np.concatenate(feature_list)

def process_sample(sample_info, use_original, use_inverted, feature_config):
    path, obj_type = sample_info
    base = os.path.basename(path)
    gi_file = os.path.join(path, f"{base}_GI_specImg.fits")
    gv_file = os.path.join(path, f"{base}_GV_specImg.fits")
    if not (os.path.exists(gi_file) and os.path.exists(gv_file)):
        print(f"跳过 {path}: 缺少 GI 或 GV 文件")
        return None
    try:
        feature_parts = []
        if use_original:
            feat_gi_orig = extract_features(gi_file, invert=False, feature_config=feature_config)
            feat_gv_orig = extract_features(gv_file, invert=False, feature_config=feature_config)
            feature_parts.extend([feat_gi_orig, feat_gv_orig])
        if use_inverted:
            feat_gi_inv = extract_features(gi_file, invert=True, feature_config=feature_config)
            feat_gv_inv = extract_features(gv_file, invert=True, feature_config=feature_config)
            feature_parts.extend([feat_gi_inv, feat_gv_inv])
        if not feature_parts:
            return None
        combined = np.concatenate(feature_parts)
        label = 1 if obj_type == 'galaxy' else 0
        return (combined, label, path)   # 返回特征、标签、路径
    except Exception as e:
        print(f"处理 {path} 时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Extract topological features and save to .npz")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores")
    parser.add_argument("--use-original", action="store_true", default=True)
    parser.add_argument("--no-original", dest="use_original", action="store_false")
    parser.add_argument("--use-inverted", action="store_true", default=True)
    parser.add_argument("--no-inverted", dest="use_inverted", action="store_false")
    parser.add_argument("--features", nargs='+', choices=['betti', 'entropy', 'landscape', 'amplitude'],
                        default=['betti', 'entropy'], help="Topological features to use")
    parser.add_argument("--betti-bins", type=int, default=20)
    parser.add_argument("--landscape-layers", type=int, default=2)
    parser.add_argument("--landscape-bins", type=int, default=50)
    parser.add_argument("--amplitude-metric", type=str, default='bottleneck',
                        choices=['bottleneck', 'wasserstein', 'heat'])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit the number of samples per class (for testing)")
    args = parser.parse_args()

    feature_config = {
        'betti': 'betti' in args.features,
        'betti_bins': args.betti_bins,
        'entropy': 'entropy' in args.features,
        'landscape': 'landscape' in args.features,
        'landscape_n_layers': args.landscape_layers,
        'landscape_n_bins': args.landscape_bins,
        'amplitude': 'amplitude' in args.features,
        'amplitude_metric': args.amplitude_metric,
        'amplitude_params': {},
    }
    print("特征配置:", feature_config)

    num_workers = args.cores if args.cores is not None else mp.cpu_count()
    print(f"使用 {num_workers} 个进程")

    data_root = "output_challenge"
    gal_dir = os.path.join(data_root, "galaxies")
    star_dir = os.path.join(data_root, "stars")

    all_gal = sorted(glob.glob(os.path.join(gal_dir, "gal_*")))
    all_star = sorted(glob.glob(os.path.join(star_dir, "star_*")))

    if args.max_samples:
        all_gal = all_gal[:args.max_samples]
        all_star = all_star[:args.max_samples]

    tasks = [(p, 'galaxy') for p in all_gal] + [(p, 'star') for p in all_star]
    print(f"总任务数: {len(tasks)}")

    process_func = partial(process_sample,
                           use_original=args.use_original,
                           use_inverted=args.use_inverted,
                           feature_config=feature_config)

    results = []
    with mp.Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_func, tasks), total=len(tasks)):
            if res is not None:
                results.append(res)

    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    paths = np.array([r[2] for r in results])

    print(f"特征矩阵形状: {X.shape}, 星系: {np.sum(y==1)}, 恒星: {np.sum(y==0)}")

    # 保存为压缩的 .npz 文件
    np.savez('tda_features.npz', X=X, y=y, paths=paths)
    print("特征已保存到 tda_features.npz")

if __name__ == "__main__":
    main()

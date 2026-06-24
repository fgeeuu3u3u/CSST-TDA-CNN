##########  python cal_snr.py

import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

SNR_CACHE_FILE = 'snr_values.npz'

def compute_snr_from_tab(fits_path):
    """计算信噪比"""
    try:
        with fits.open(fits_path) as hdul:
            for hdu in hdul:
                if hasattr(hdu, 'columns') and 'FLUX' in hdu.columns.names:
                    flux = hdu.data['FLUX'].astype(np.float64)
                    err = hdu.data['ERR'].astype(np.float64)
                    break
            else:
                return np.nan
    except Exception:
        return np.nan
    flux[flux < 0] = 0.0
    err[err <= 0] = np.nan
    ratio = flux / err
    valid = np.isfinite(ratio)
    if np.sum(valid) == 0:
        return 0.0
    return np.nanmean(ratio[valid])

def collect_snr(base_dir='../output_challenge', band='GI'):
    """
    收集所有恒星和星系的 SNR 值及对应的样本目录路径。
    返回: (snr_stars, snr_galaxies, paths_stars, paths_galaxies)
    路径格式: "output_challenge/galaxies/gal_0" 或 "output_challenge/stars/star_0"
    """
    snr_stars = []
    snr_galaxies = []
    paths_stars = []
    paths_galaxies = []
    
    # 星系
    gal_dir = os.path.join(base_dir, 'galaxies')
    if os.path.exists(gal_dir):
        for gal_id in tqdm(os.listdir(gal_dir), desc="星系"):
            gal_path = os.path.join(gal_dir, gal_id)
            if not os.path.isdir(gal_path):
                continue
            tab_file = f"{gal_id}_{band}_specTab.fits"
            tab_path = os.path.join(gal_path, tab_file)
            if os.path.exists(tab_path):
                snr = compute_snr_from_tab(tab_path)
                if not np.isnan(snr):
                    snr_galaxies.append(snr)
                    paths_galaxies.append(os.path.join('output_challenge', 'galaxies', gal_id))
    
    # 恒星
    star_dir = os.path.join(base_dir, 'stars')
    if os.path.exists(star_dir):
        for star_id in tqdm(os.listdir(star_dir), desc="恒星"):
            star_path = os.path.join(star_dir, star_id)
            if not os.path.isdir(star_path):
                continue
            tab_file = f"{star_id}_{band}_specTab.fits"
            tab_path = os.path.join(star_path, tab_file)
            if os.path.exists(tab_path):
                snr = compute_snr_from_tab(tab_path)
                if not np.isnan(snr):
                    snr_stars.append(snr)
                    paths_stars.append(os.path.join('output_challenge', 'stars', star_id))
    
    return np.array(snr_stars), np.array(snr_galaxies), np.array(paths_stars), np.array(paths_galaxies)

if __name__ == "__main__":
    if os.path.exists(SNR_CACHE_FILE):
        print(f"发现缓存文件 {SNR_CACHE_FILE}，直接加载...")
        data = np.load(SNR_CACHE_FILE, allow_pickle=True)
        snr_stars = data['snr_stars']
        snr_galaxies = data['snr_galaxies']
        paths_stars = data['paths_stars']
        paths_galaxies = data['paths_galaxies']
    else:
        print("未找到缓存文件，正在计算所有样本的 SNR...")
        snr_stars, snr_galaxies, paths_stars, paths_galaxies = collect_snr(base_dir='../output_challenge', band='GI')
        np.savez(SNR_CACHE_FILE, 
                 snr_stars=snr_stars, 
                 snr_galaxies=snr_galaxies,
                 paths_stars=paths_stars,
                 paths_galaxies=paths_galaxies)
        print(f"SNR 数据及路径已保存到 {SNR_CACHE_FILE}")
    
    print(f"恒星样本数: {len(snr_stars)}, 星系样本数: {len(snr_galaxies)}")
    if len(snr_stars) > 0:
        print(f"恒星 SNR: 最小 {np.min(snr_stars):.2f}, 最大 {np.max(snr_stars):.2f}, 中位数 {np.median(snr_stars):.2f}")
    if len(snr_galaxies) > 0:
        print(f"星系 SNR: 最小 {np.min(snr_galaxies):.2f}, 最大 {np.max(snr_galaxies):.2f}, 中位数 {np.median(snr_galaxies):.2f}")
    
    # 打印路径示例
    if len(paths_stars) > 0:
        print(f"示例恒星路径: {paths_stars[0]}")
    if len(paths_galaxies) > 0:
        print(f"示例星系路径: {paths_galaxies[0]}")
# 使用默认阈值 (re>6.0 或 n>5.0 为大星系)
#   python generate_challenge_spectra.py --cores 16

# 自定义阈值（例如 re>5.0 或 n>4.0）
#   python generate_challenge_spectra.py --cores 16 --large-re 5.0 --large-n 4.0

# 禁用大星系单独处理（将所有星系视为正常，即用并行池）
#   python generate_challenge_spectra.py --cores 16 --large-re 100 --large-n 100

#!/usr/bin/env python3

"""
生成挑战赛光谱（支持大星系单独处理）
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from scipy import interpolate
import galsim
from SpecGen.SpecGenerator import SpecGenerator
from SpecGen.Config import Config
import warnings
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings("ignore")

# ================= 配置参数 =================
BASE_DIR = "csst_slitless_classification"
STAR_DIR = os.path.join(BASE_DIR, "mock_stars")
GAL_DIR = os.path.join(BASE_DIR, "mock_galaxies")

BANDS = ['GI', 'GV']
BEAM = 'A'
APER = 2.0
P_SIZE = 0.074
EXP_TIME = 150.0
EXP_NUM = 1
READOUT = 5.0
DARK = 0.02
SKYBG_DICT = {'GU': 0.019, 'GV': 0.214, 'GI': 0.329}
PSF_FWHM = 0.15
psf = galsim.Gaussian(fwhm=PSF_FWHM)

OUTPUT_BASE = "./output_challenge"
os.makedirs(OUTPUT_BASE, exist_ok=True)

dataDir = '../data/'
config = Config(dataDir=dataDir)

# ================= 辅助函数 =================
def uJy_to_erg_s_cm2_A(wave_A, flux_uJy):
    c_A = 2.99792458e18
    return flux_uJy * 1e-29 * c_A / wave_A**2

def expand_sed_to_csst_grid(wave, flux):
    new_wave = np.arange(2000, 11000 + 0.5, 0.5)
    f = interpolate.interp1d(wave, flux, kind='linear', fill_value=0.0, bounds_error=False)
    return new_wave, f(new_wave)

def save_sed_to_temp(wave, flux, prefix="sed"):
    temp_dir = "./temp_sed"
    os.makedirs(temp_dir, exist_ok=True)
    import random
    temp_file = os.path.join(temp_dir, f"{prefix}_{random.randint(0,1e9)}.txt")
    np.savetxt(temp_file, np.column_stack((wave, flux)), fmt='%.6e')
    return temp_file

def e1e2_to_q_pa(e1, e2):
    e = np.hypot(e1, e2)
    if e >= 1.0:
        e = 0.999
    q = (1.0 - e) / (1.0 + e)
    pa = 0.5 * np.arctan2(e2, e1) * 180.0 / np.pi
    return q, pa

def crop_center(img, crop_size):
    if isinstance(crop_size, int):
        crop_height = crop_width = crop_size
    else:
        crop_height, crop_width = crop_size
    height, width = img.shape
    start_y = max(0, (height - crop_height) // 2)
    start_x = max(0, (width - crop_width) // 2)
    end_y = min(height, start_y + crop_height)
    end_x = min(width, start_x + crop_width)
    return img[start_y:end_y, start_x:end_x]

def padding_img(img, xsize, ysize, skybg, dark, readout, expTime, expNum, seed=1234):
    xsize = max(xsize, img.shape[1])
    ysize = max(ysize, img.shape[0])
    mean = (skybg + dark) * expTime * expNum
    np.random.seed(seed)
    container = np.random.poisson(np.full((ysize, xsize), mean))
    for i in range(expNum):
        np.random.seed(seed + i * seed)
        container += np.around(np.random.normal(0, readout, container.shape), 0).astype(int)
    container = container.astype(np.float32) - mean
    spec_img_height, spec_img_width = img.shape
    start_row = max(0, (ysize - spec_img_height) // 2)
    start_col = max(0, (xsize - spec_img_width) // 2)
    end_row = min(ysize, start_row + spec_img_height)
    end_col = min(xsize, start_col + spec_img_width)
    img_height_to_use = end_row - start_row
    img_width_to_use = end_col - start_col
    container[start_row:end_row, start_col:end_col] = img[:img_height_to_use, :img_width_to_use]
    return container

def standardize_img(img, xsize, ysize, skybg, dark, readout, expTime, expNum, seed=1234):
    pad_img = padding_img(img, xsize, ysize, skybg, dark, readout, expTime, expNum, seed=seed)
    crop_img = crop_center(pad_img, (ysize, xsize))
    return crop_img

# ================= 加载数据 =================
print("Loading catalogs...")
star_meta = Table.read(os.path.join(STAR_DIR, "star_mata.fits"))
gal_meta = Table.read(os.path.join(GAL_DIR, "galaxy_meta.fits"))

print("Reading spectra...")
star_hdu = fits.open(os.path.join(STAR_DIR, "input-spectra.fits"))
star_wave = star_hdu[0].data
star_flux_all = star_hdu[1].data

gal_hdu = fits.open(os.path.join(GAL_DIR, "galaxy-spectra.uJy.fits"))
band_map = {'GU':(0,1), 'GV':(2,3), 'GI':(4,5)}
gal_wave_cache = {}
gal_flux_cache = {}
for band in BANDS:
    wave_hdu, flux_hdu = band_map[band]
    gal_wave_cache[band] = gal_hdu[wave_hdu].data
    gal_flux_cache[band] = gal_hdu[flux_hdu].data

TARGET_WIDTH = 768
TARGET_HEIGHT = 128
TARGET_IMG_SIZE = 128
SEED = 1234

# ================= 处理函数 =================
def process_single_galaxy(args):
    idx, row = args
    z = row['REDSHIFT']
    sersic_n = row['SERSIC']
    re = row['SHAPE_R']
    e1 = row['SHAPE_E1']
    e2 = row['SHAPE_E2']
    q_ell, pa = e1e2_to_q_pa(e1, e2)
    src_dir = os.path.join(gal_out_dir, f"gal_{idx}")
    os.makedirs(src_dir, exist_ok=True)
    for band in BANDS:
        ext = row.get({'GU':'EXT_G','GV':'EXT_R','GI':'EXT_Z'}[band], 0.0)
        skybg = SKYBG_DICT[band]
        gal_wave = gal_wave_cache[band]
        gal_flux_all = gal_flux_cache[band]
        flux_int = gal_flux_all[idx, :]
        wave_obs = gal_wave * (1.0 + z)
        flux_obs = flux_int * 10**(-0.4 * ext)
        flux_obs_erg = uJy_to_erg_s_cm2_A(wave_obs, flux_obs)
        wave_grid, flux_grid = expand_sed_to_csst_grid(wave_obs, flux_obs_erg)
        temp_sed = save_sed_to_temp(wave_grid, flux_grid, f"gal_{idx}_{band}")
        specG = SpecGenerator(sedFn=temp_sed, grating=band, beam=BEAM, aper=APER,
                              xcenter=100., ycenter=100., p_size=P_SIZE, psf=psf,
                              skybg=skybg, dark=DARK, readout=READOUT, t=EXP_TIME,
                              expNum=EXP_NUM, config=config)
        try:
            specTab, specImg, img, satPix = specG.generateSpec1dforGal(
                s_n=sersic_n, re=re, pa=pa, q_ell=q_ell,
                limitfluxratio=0.8, deltLamb=0.01
            )
            specImg = standardize_img(specImg, TARGET_WIDTH, TARGET_HEIGHT,
                                      skybg, DARK, READOUT, EXP_TIME, EXP_NUM, seed=SEED)
            img = crop_center(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        except Exception as e:
            print(f"\nGalaxy {idx} band {band} failed: {e}")
            os.remove(temp_sed) if os.path.exists(temp_sed) else None
            continue
        out_prefix = os.path.join(src_dir, f"gal_{idx}_{band}")
        fits.writeto(out_prefix + "_specImg.fits", specImg, overwrite=True)
        fits.writeto(out_prefix + "_DImg.fits", img, overwrite=True)
        specTab.write(out_prefix + "_specTab.fits", overwrite=True)
        os.remove(temp_sed)
    return idx

def process_single_star(args):
    idx, row = args
    tid = row['TARGETID']
    flux_int = star_flux_all[idx, :] * 1e-17
    _, flux_grid = expand_sed_to_csst_grid(star_wave, flux_int)
    src_dir = os.path.join(star_out_dir, f"star_{tid}")
    os.makedirs(src_dir, exist_ok=True)
    for band in BANDS:
        skybg = SKYBG_DICT[band]
        temp_sed = save_sed_to_temp(star_wave_grid, flux_grid, f"star_{tid}_{band}")
        specG = SpecGenerator(sedFn=temp_sed, grating=band, beam=BEAM, aper=APER,
                              xcenter=100., ycenter=100., p_size=P_SIZE, psf=psf,
                              skybg=skybg, dark=DARK, readout=READOUT, t=EXP_TIME,
                              expNum=EXP_NUM, config=config)
        try:
            specTab, specImg, img, satPix = specG.generateSpec1dforStar(limitfluxratio=0.8)
            specImg = standardize_img(specImg, TARGET_WIDTH, TARGET_HEIGHT,
                                      skybg, DARK, READOUT, EXP_TIME, EXP_NUM, seed=SEED)
            img = crop_center(img, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        except Exception as e:
            print(f"\nStar {tid} band {band} failed: {e}")
            os.remove(temp_sed) if os.path.exists(temp_sed) else None
            continue
        out_prefix = os.path.join(src_dir, f"star_{tid}_{band}")
        fits.writeto(out_prefix + "_specImg.fits", specImg, overwrite=True)
        fits.writeto(out_prefix + "_DImg.fits", img, overwrite=True)
        specTab.write(out_prefix + "_specTab.fits", overwrite=True)
        os.remove(temp_sed)
    return idx


def main():
    parser = argparse.ArgumentParser(description="Generate challenge spectra with multiprocessing")
    parser.add_argument("--cores", type=int, default=None,
                        help="Number of CPU cores to use (default: all available)")
    parser.add_argument("--large-re", type=float, default=6.0,
                        help="Half-light radius threshold for large galaxies (arcsec)")
    parser.add_argument("--large-n", type=float, default=5.0,
                        help="Sersic index threshold for large galaxies")
    args = parser.parse_args()
    num_workers = args.cores if args.cores else os.cpu_count()
    print(f"Using {num_workers} CPU cores for normal galaxies and stars")
    print(f"Large galaxy threshold: re > {args.large_re} or n > {args.large_n}")

    # 准备任务列表
    gal_tasks = [(idx, row) for idx, row in enumerate(gal_meta)]
    star_tasks = [(idx, row) for idx, row in enumerate(star_meta)]

    # 根据阈值分割星系任务
    normal_gal_tasks = []
    large_gal_tasks = []
    for idx, row in enumerate(gal_meta):
        re = row['SHAPE_R']
        n = row['SERSIC']
        if re > args.large_re or n > args.large_n:
            large_gal_tasks.append((idx, row))
        else:
            normal_gal_tasks.append((idx, row))
    print(f"Normal galaxies: {len(normal_gal_tasks)}, Large galaxies: {len(large_gal_tasks)}")

    # 设置全局输出目录
    global gal_out_dir, star_out_dir, star_wave_grid
    gal_out_dir = os.path.join(OUTPUT_BASE, "galaxies")
    star_out_dir = os.path.join(OUTPUT_BASE, "stars")
    star_wave_grid = np.arange(2000, 11000 + 0.5, 0.5)
    os.makedirs(gal_out_dir, exist_ok=True)
    os.makedirs(star_out_dir, exist_ok=True)

    # ========== 顺序调整 ==========
    # 1. 先处理正常星系（多进程，使用全部核心）
    if normal_gal_tasks:
        print(f"\nSimulating normal galaxies for bands {BANDS} with {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_galaxy, normal_gal_tasks),
                                total=len(normal_gal_tasks), desc="Normal Galaxies"))

    # 2. 再处理恒星（多进程，使用全部核心）
    if star_tasks:
        print(f"\nSimulating stars for bands {BANDS} with {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_star, star_tasks),
                                total=len(star_tasks), desc="Stars"))

    # 3. 最后处理大星系（多进程避免内存爆炸，固定使用 6 个进程）
    if large_gal_tasks:
        large_workers = 6  
        print(f"\nSimulating large galaxies for bands {BANDS} with {large_workers} workers (one by one in parallel)...")
        with mp.Pool(processes=large_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_galaxy, large_gal_tasks),
                                total=len(large_gal_tasks), desc="Large Galaxies"))

    print(f"\nAll done! Results saved in: {OUTPUT_BASE}")

if __name__ == "__main__":
    main()

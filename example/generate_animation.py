'''
使用方法

# 生成星系0的GI波段动画（默认取反，左右同步）
python generate_animation.py --type galaxy --idx 0 --band GI --crop 128

# 如果不想要取反（显示原始山峰），加上 --no-invert
python generate_animation.py --type galaxy --idx 0 --band GI --crop 12 --no-invert

# 生成恒星86的GV波段动画，下采样2倍并裁剪中心64x64
python generate_animation.py --type star --idx 86 --band GV --downsample 2 --crop 64

# 自定义帧数
python generate_animation.py --type galaxy --idx 1026 --band GI --n-frames 20 --crop 20
'''

import os
import argparse
import numpy as np
from astropy.io import fits
from gtda.homology import CubicalPersistence
from gtda.diagrams import Scaler, BettiCurve
from gtda.plotting import plot_diagram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

# ==============================================================
# 持久图计算和绘图函数
# ==============================================================
def compute_persistence_diagram(fits_path, title=None):
    """
    与静态脚本完全相同的持久图计算和绘图。
    返回 diag_scaled 用于后续动画。
    """
    img = fits.getdata(fits_path)
    img_norm = (img - img.min()) / (img.max() - img.min())
    X = img_norm[np.newaxis, :, :]

    cubical = CubicalPersistence(homology_dimensions=(0, 1), n_jobs=1)
    diagrams = cubical.fit_transform(X)
    scaler = Scaler()
    diag_scaled = scaler.fit_transform(diagrams)[0]

    # 1. 交互式 HTML（与静态脚本一致）
    fig1 = plot_diagram(diag_scaled)
    html_path = f"{title}_persistence.html"
    fig1.write_html(html_path)
    print(f"交互式持久图已保存: {html_path}")

    # 2. 静态 PNG（与静态脚本一致）
    h0 = diag_scaled[diag_scaled[:, 2] == 0]
    h1 = diag_scaled[diag_scaled[:, 2] == 1]
    plt.figure(figsize=(6, 6))
    plt.scatter(h0[:, 0], h0[:, 1], c='red', s=50, alpha=0.7, label='H0')
    if len(h1) > 0:
        plt.scatter(h1[:, 0], h1[:, 1], c='blue', s=50, alpha=0.7, label='H1')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    plt.xlabel('Birth time')
    plt.ylabel('Death time')
    plt.title(f'Persistence Diagram - {title}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    png_path = f"{title}_persistence.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"静态持久图已保存: {png_path}")

    # 3. Betti曲线（可选，但保留与静态脚本一致）
    betti = BettiCurve(n_bins=100, n_jobs=1)
    betti_curves = betti.fit_transform(diag_scaled[np.newaxis, :, :])[0]
    thresholds = np.linspace(0, 1, betti_curves.shape[1])

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, betti_curves[0], label='H0 (Connected components)', linewidth=2)
    plt.plot(thresholds, betti_curves[1], label='H1 (Holes)', linewidth=2)
    plt.xlabel('Filtration threshold')
    plt.ylabel('Betti number')
    plt.title(f'Betti curves - {title}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    betti_png = f"{title}_betti_curves.png"
    plt.savefig(betti_png, dpi=150)
    plt.close()
    print(f"Betti曲线图已保存: {betti_png}")

    return diag_scaled



# ==============================================================
def create_levelset_persistence_animation(fits_path, output_gif='animation.gif', 
                                          n_frames=30, downsample_factor=1, 
                                          crop_size=None, invert=True):
    """
    生成二维光谱水平集过滤 + 条形码联动的GIF动画。
    """
    # ---- 读取原始图像（用于持久图计算，与静态脚本共用） ----
    img_original = fits.getdata(fits_path)

    # ---- 生成与静态脚本完全一致的持久图数据 ----
    base_title = os.path.splitext(os.path.basename(fits_path))[0]
    diag_scaled = compute_persistence_diagram(fits_path, title=base_title)

    # ---- 分离 H0 和 H1 ----
    h0 = diag_scaled[diag_scaled[:, 2] == 0]
    h1 = diag_scaled[diag_scaled[:, 2] == 1]

    # 按出生时间排序
    h0 = h0[np.argsort(h0[:, 0])]
    h1 = h1[np.argsort(h1[:, 0])]

    # ---- 显示图像预处理 ----
    img_display = img_original.copy()
    if invert:
        img_display = 1 - img_display
        print("已启用取反：图像翻转，左图将显示低值区域（原山峰）")
    if downsample_factor > 1:
        img_display = img_display[::downsample_factor, ::downsample_factor]
    if crop_size is not None:
        h, w = img_display.shape
        if crop_size > min(h, w):
            print(f"警告: crop_size ({crop_size}) 大于图像最小尺寸 ({min(h,w)})，将使用原尺寸")
        else:
            cy, cx = h // 2, w // 2
            half = crop_size // 2
            img_display = img_display[cy-half:cy+half, cx-half:cx+half]
            print(f"裁剪中心 {crop_size}x{crop_size} 区域，新尺寸: {img_display.shape}")

    img_display_norm = (img_display - img_display.min()) / (img_display.max() - img_display.min())

    # ---- 生成动画 ----
    thresholds = np.linspace(0, 1, n_frames)
    frames = []
    print(f"生成 {n_frames} 帧动画...")

    n0 = len(h0)
    n1 = len(h1)
    y0 = np.arange(n0)
    y1 = np.arange(n1) + n0 + 1

    for thresh in tqdm(thresholds, desc="渲染帧"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ---------- 左图：水平集 ----------
        masked = np.where(img_display_norm < thresh, img_display_norm, np.nan)
        ax1.imshow(img_display_norm, cmap='gray', alpha=0.3, origin='lower')
        im = ax1.imshow(masked, cmap='viridis', origin='lower', alpha=0.9)
        ax1.set_title(f'Level Set: threshold = {thresh:.3f}')
        ax1.set_xlabel('Wavelength (pixel)')
        ax1.set_ylabel('Spatial (pixel)')
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # ---------- 右图：条形码 ----------
        # 绘制 H0 (红色)
        for i, (birth, death, _) in enumerate(h0):
            if birth <= thresh:
                end = min(death, thresh)
                ax2.hlines(y=y0[i], xmin=birth, xmax=end,
                           colors='red', linewidth=3, alpha=0.8)
        # 绘制 H1 (蓝色)
        for i, (birth, death, _) in enumerate(h1):
            if birth <= thresh:
                end = min(death, thresh)
                ax2.hlines(y=y1[i], xmin=birth, xmax=end,
                           colors='blue', linewidth=3, alpha=0.8)

        # 装饰条形码
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.5, max(y1) + 1 if n1 > 0 else n0 + 1)
        ax2.set_xlabel('Filtration threshold')
        ax2.set_ylabel('')
        ax2.set_yticks([])
        ax2.set_title('')          # 去掉标题
        ax2.grid(alpha=0.3, axis='x')

        # 添加 dim0 / dim1 标签（放置在纵轴左侧）
        from matplotlib.patches import Patch
        if n0 > 0:
            y_center0 = (n0 - 1) / 2
            ax2.text(-0.06, y_center0, 'dim0', ha='right', va='center',
                     fontsize=12, color='red', transform=ax2.get_xaxis_transform())
        if n1 > 0:
            y_center1 = n0 + 1 + (n1 - 1) / 2
            ax2.text(-0.06, y_center1, 'dim1', ha='right', va='center',
                     fontsize=12, color='blue', transform=ax2.get_xaxis_transform())

        # 图例
        legend_elements = [Patch(facecolor='red', label='H0'),
                           Patch(facecolor='blue', label='H1')]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)

        # ---- 布局调整：用 subplots_adjust 替代 tight_layout ----
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.9)

        # 保存帧
        fig.canvas.draw()
        buf, (width, height) = fig.canvas.print_to_buffer()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(output_gif, frames, fps=5, loop=0)
    print("动画生成完成！")

def main():
    parser = argparse.ArgumentParser(description="生成CSST二维光谱的水平集过滤+持久图联动动画")
    parser.add_argument('--type', choices=['galaxy', 'star'], required=True,
                        help="天体类型：galaxy 或 star")
    parser.add_argument('--idx', type=int, required=True,
                        help="天体编号")
    parser.add_argument('--band', choices=['GI', 'GV'], default=None,
                        help="波段：GI 或 GV，不指定则同时生成两个")
    parser.add_argument('--n-frames', type=int, default=30,
                        help="动画帧数（默认30）")
    parser.add_argument('--downsample', type=int, default=1,
                        help="下采样因子（默认1，大于1可加速）")
    parser.add_argument('--crop', type=int, default=None,
                        help="裁剪图像中心区域的大小（像素），例如 --crop 128，显示中心区域细节")
    parser.add_argument('--invert', action='store_true', default=True,
                        help="是否对图像取反（默认启用），使左图与右图同步")
    parser.add_argument('--no-invert', action='store_false', dest='invert',
                        help="禁用取反，显示原始图像（山峰）")
    parser.add_argument('--base-dir', type=str, default='output_challenge',
                        help="数据根目录（默认当前目录下的 output_challenge）")
    parser.add_argument('--output-dir', type=str, default='.',
                        help="输出GIF的目录（默认当前目录）")

    args = parser.parse_args()

    if args.type == 'galaxy':
        prefix = 'gal'
        dir_template = f"{args.base_dir}/galaxies/gal_{{idx}}"
        file_template = f"gal_{{idx}}_{{band}}_specImg.fits"
    else:
        prefix = 'star'
        dir_template = f"{args.base_dir}/stars/star_{{idx}}"
        file_template = f"star_{{idx}}_{{band}}_specImg.fits"

    bands = [args.band] if args.band else ['GI', 'GV']

    for band in bands:
        dir_path = dir_template.format(idx=args.idx)
        file_name = file_template.format(idx=args.idx, band=band)
        fits_path = os.path.join(dir_path, file_name)

        if not os.path.exists(fits_path):
            print(f"警告：文件不存在 {fits_path}，跳过")
            continue

        output_gif = os.path.join(args.output_dir, f"{prefix}_{args.idx}_{band}_animation.gif")

        print(f"\n处理 {args.type} {args.idx}, 波段 {band}")
        print(f"输入文件: {fits_path}")
        print(f"输出文件: {output_gif}")
        if args.crop:
            print(f"裁剪中心区域: {args.crop}x{args.crop}")
        print(f"取反: {'启用' if args.invert else '禁用'}")

        create_levelset_persistence_animation(
            fits_path,
            output_gif=output_gif,
            n_frames=args.n_frames,
            downsample_factor=args.downsample,
            crop_size=args.crop,
            invert=args.invert
        )

if __name__ == "__main__":
    main()

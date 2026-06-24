##########  python plot_snr_train_test.py
import os
import numpy as np
import matplotlib.pyplot as plt

SNR_CACHE = 'snr_values.npz'
TEST_PATHS_FILE = '../test_paths.txt'
MISCLASSIFIED_PATHS_FILE = '../misclassified_paths.txt'

def load_snr_data():
    """加载信噪比缓存数据"""
    data = np.load(SNR_CACHE, allow_pickle=True)
    return data['snr_stars'], data['snr_galaxies'], data['paths_stars'], data['paths_galaxies']

def load_paths(filename):
    """读取路径列表"""
    if not os.path.exists(filename):
        print(f"警告：文件 {filename} 不存在，跳过。")
        return []
    with open(filename, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths

def normalize_path(path):
    """
    统一路径格式，用于匹配。
    最终路径形如 'output_challenge/galaxies/gal_0'
    """
    # 去除开头可能存在的 './' 或 '../'
    if path.startswith('./'):
        path = path[2:]
    if path.startswith('../'):
        if path.startswith('../output_challenge/'):
            path = path[3:]  # 去掉 '../'
        else:
            # 其他情况保留，但后面会尝试添加前缀
            pass
    if not path.startswith('output_challenge/'):
        if path.startswith('galaxies/') or path.startswith('stars/'):
            path = 'output_challenge/' + path
    return path

def get_misclassified_snr(paths_stars, paths_galaxies, snr_stars, snr_galaxies, mis_paths):
    """
    根据错误路径列表，从SNR数据中提取对应的SNR值。
    返回: (mis_stars_snr, mis_galaxies_snr)
    """
    mis_set = set(normalize_path(p) for p in mis_paths)
    mis_stars = []
    mis_galaxies = []
    # 匹配恒星
    for p, snr in zip(paths_stars, snr_stars):
        if normalize_path(p) in mis_set:
            mis_stars.append(snr)
    # 匹配星系
    for p, snr in zip(paths_galaxies, snr_galaxies):
        if normalize_path(p) in mis_set:
            mis_galaxies.append(snr)
    return np.array(mis_stars), np.array(mis_galaxies)

def plot_snr_distribution(snr_stars, snr_galaxies, test_paths,
                          mis_stars_snr, mis_galaxies_snr):
    """
    绘制训练集、测试集和错误样本的SNR分布阶梯图。
    训练集: 实线，测试集: 虚线，错误样本: 点划线(dashdot) + 标记'o'
    """
    # 标准化测试集路径
    test_paths_set = set(normalize_path(p) for p in test_paths)
    
    pass

# 主流程
def main():
    # 加载SNR数据
    if not os.path.exists(SNR_CACHE):
        print(f"错误：未找到 {SNR_CACHE}，请先运行计算SNR的脚本。")
        return
    snr_stars, snr_galaxies, paths_stars, paths_galaxies = load_snr_data()
    
    # 加载测试集路径
    test_paths = load_paths(TEST_PATHS_FILE)
    if not test_paths:
        print("未找到测试集路径，无法区分训练/测试。")
        return
    test_paths_set = set(normalize_path(p) for p in test_paths)
    
    # 加载错误样本路径
    mis_paths = load_paths(MISCLASSIFIED_PATHS_FILE)
    if mis_paths:
        mis_set = set(normalize_path(p) for p in mis_paths)
    else:
        mis_set = set()
    
    # 划分训练/测试集SNR
    train_stars, test_stars = [], []
    train_galaxies, test_galaxies = [], []
    mis_stars, mis_galaxies = [], []
    
    for p, snr in zip(paths_stars, snr_stars):
        norm_p = normalize_path(p)
        if norm_p in test_paths_set:
            test_stars.append(snr)
        else:
            train_stars.append(snr)
        if norm_p in mis_set:
            mis_stars.append(snr)
    
    for p, snr in zip(paths_galaxies, snr_galaxies):
        norm_p = normalize_path(p)
        if norm_p in test_paths_set:
            test_galaxies.append(snr)
        else:
            train_galaxies.append(snr)
        if norm_p in mis_set:
            mis_galaxies.append(snr)
    
    # 转为数组
    train_stars = np.array(train_stars)
    test_stars = np.array(test_stars)
    train_galaxies = np.array(train_galaxies)
    test_galaxies = np.array(test_galaxies)
    mis_stars = np.array(mis_stars)
    mis_galaxies = np.array(mis_galaxies)
    
    print(f"训练集恒星: {len(train_stars)}, 测试集恒星: {len(test_stars)}")
    print(f"训练集星系: {len(train_galaxies)}, 测试集星系: {len(test_galaxies)}")
    print(f"错误恒星: {len(mis_stars)}, 错误星系: {len(mis_galaxies)}")
    
    # 合并所有SNR确定上限
    all_snr = np.concatenate([train_stars, test_stars, train_galaxies, test_galaxies,
                              mis_stars, mis_galaxies])
    if len(all_snr) == 0:
        print("没有有效的SNR数据！")
        return
    upper_limit = np.percentile(all_snr, 99)
    max_snr = upper_limit * 1.05
    bins = np.linspace(0, max_snr, 60)
    
    plt.figure(figsize=(10, 6))
    
    # 恒星：红色
    plt.hist(train_stars, bins=bins, histtype='step', linewidth=2, linestyle='solid',
             label='Train Stars', color='red')
    plt.hist(test_stars, bins=bins, histtype='step', linewidth=2, linestyle='dashed',
             label='Test Stars', color='red')
    # 错误恒星：填充样式
    plt.hist(mis_stars, bins=bins, histtype='stepfilled', alpha=0.5,
             label='Misclassified Stars', color='darkred')
    
    # 星系：蓝色
    plt.hist(train_galaxies, bins=bins, histtype='step', linewidth=2, linestyle='solid',
             label='Train Galaxies', color='blue')
    plt.hist(test_galaxies, bins=bins, histtype='step', linewidth=2, linestyle='dashed',
             label='Test Galaxies', color='blue')
    # 错误星系：填充样式
    plt.hist(mis_galaxies, bins=bins, histtype='stepfilled', alpha=0.5,
             label='Misclassified Galaxies', color='darkblue')
    
    plt.yscale('log')
    plt.xlabel('Signal-to-Noise Ratio (SNR)')
    plt.ylabel('Number')
    plt.title('SNR Distribution in GI band')
    plt.legend()
    plt.tight_layout()
    plt.savefig('snr_train_test_misclassified.png', dpi=150)
    plt.close()
    print("图像已保存为 snr_train_test_misclassified.png")

if __name__ == "__main__":
    main()
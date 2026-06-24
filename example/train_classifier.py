#!/usr/bin/env python3
"""
===============================================================================
星系-恒星分类器训练脚本（基于预提取的拓扑特征）

功能：
    加载由 extract_features.py 生成的 .npz 特征文件（包含特征矩阵 X、标签 y、样本路径 paths），
    可选进行随机森林超参数搜索（随机搜索或网格搜索），训练最佳模型，保存模型及配置。
    额外保存测试集路径和分类错误样本路径，便于后续分析。

使用方法：
    python train_classifier.py [选项]

核心参数：
    --feature-file PATH         特征文件路径（默认: tda_features.npz）
    --test-size FLOAT           测试集比例（默认: 0.3）
    --random-seed INT           随机种子（默认: 42）
    --search                    启用超参数搜索（默认禁用，直接使用默认参数训练）
    --search-mode {grid,random} 搜索模式：grid（穷举网格搜索）或 random（随机搜索），默认 random
    --n-iter INT                随机搜索的迭代次数（参数组合数），默认 15
    --cv-folds INT              交叉验证折数（默认: 3）
    --n-jobs INT                并行计算使用的CPU核心数（默认: -1，表示全部核心）
    --output-model PATH         输出模型文件路径（默认: star_galaxy_classifier.joblib）
    --save-test-paths           保存测试集样本路径到文件（默认: test_paths.txt）
    --smote-ratio FLOAT         BorderlineSMOTE采样后少数类目标比例（默认: 1.0，即与多数类持平）

示例：
    # 默认直接训练（不使用搜索）
    python train_classifier.py

    # 启用随机搜索（15次迭代，3折交叉验证）
    python train_classifier.py --search

    # 保存测试集和错误样本路径
    python train_classifier.py --save-test-paths --smote-ratio 0.3 --n-jobs 32

注意事项：
    - 特征文件必须包含 'X', 'y', 'paths' 三个键。
    - 训练时使用 BorderlineSMOTE 过采样处理类别不平衡。
    - 搜索启用时，交叉验证在训练集内部进行，测试集保持独立。
===============================================================================
"""
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import randint
import joblib
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="训练星系-恒星分类器（基于预提取的拓扑特征）",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--feature-file', type=str, default='tda_features.npz',
                        help='特征文件路径（默认: tda_features.npz）')
    parser.add_argument('--test-size', type=float, default=0.3,
                        help='测试集比例（默认: 0.3）')
    parser.add_argument('--random-seed', type=int, default=45,
                        help='随机种子（默认: 42）')
    parser.add_argument('--search', action='store_true',
                        help='启用超参数搜索（默认禁用，直接使用默认参数训练）')
    parser.add_argument('--search-mode', choices=['grid', 'random'], default='random',
                        help='搜索模式：grid（穷举网格搜索）或 random（随机搜索），默认 random')
    parser.add_argument('--n-iter', type=int, default=15,
                        help='随机搜索的迭代次数（参数组合数），仅在 random 模式下有效，默认 15')
    parser.add_argument('--cv-folds', type=int, default=3,
                        help='交叉验证折数（默认: 3）')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='并行计算使用的CPU核心数（默认: -1，表示全部核心）')
    parser.add_argument('--output-model', type=str, default='star_galaxy_classifier.joblib',
                        help='输出模型文件路径（默认: star_galaxy_classifier.joblib）')
    parser.add_argument('--save-test-paths', action='store_true',
                        help='保存测试集样本路径到文件（默认: test_paths.txt）')
    parser.add_argument('--smote-ratio', type=float, default=1.0,
                        help='BorderlineSMOTE采样后少数类目标比例（默认: 1.0，即与多数类持平）')
    args = parser.parse_args()

    # 1. 加载特征数据
    data = np.load(args.feature_file)
    X = data['X']
    y = data['y']
    paths = data['paths']
    print(f"加载特征矩阵: {X.shape}, 星系: {np.sum(y==1)}, 恒星: {np.sum(y==0)}")

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, paths, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    # 3. 保存测试集路径
    if args.save_test_paths:
        with open('test_paths.txt', 'w') as f:
            for p in paths_test:
                f.write(p + '\n')
        print(f"测试集路径已保存到 test_paths.txt，共 {len(paths_test)} 个样本")

    # 4. 构建 Pipeline（包含 BorderlineSMOTE 过采样）
    pipeline = Pipeline([
        ('smote', BorderlineSMOTE(random_state=args.random_seed, sampling_strategy=args.smote_ratio)),
        ('rf', RandomForestClassifier(random_state=args.random_seed, n_jobs=1, class_weight='balanced'))
    ])

    # 5. 训练模型（根据 --search 决定是否搜索）
    if args.search:
        if args.search_mode == 'random':
            # 随机搜索
            param_dist = {
                'rf__n_estimators': randint(80, 400),
                'rf__max_depth': [None, 10, 20, 30],
                'rf__min_samples_split': randint(2, 11),
            }
            print(f"开始随机搜索（评分: f1_macro，使用 {args.n_jobs} 个核心，"
                  f"{args.cv_folds} 折交叉验证，{args.n_iter} 次迭代）...")
            search = RandomizedSearchCV(
                pipeline, param_dist, n_iter=args.n_iter, cv=args.cv_folds,
                scoring='f1_macro', n_jobs=args.n_jobs,
                random_state=args.random_seed, verbose=1
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            print(f"最佳参数: {search.best_params_}")
            print(f"最佳交叉验证 F1-macro: {search.best_score_:.4f}")
        else:
            # 网格搜索
            param_grid = {
                'rf__n_estimators': [100, 200, 300],
                'rf__max_depth': [None, 10, 20],
                'rf__min_samples_split': [2, 5],
            }
            print(f"开始网格搜索（评分: f1_macro，使用 {args.n_jobs} 个核心，"
                  f"{args.cv_folds} 折交叉验证）...")
            search = GridSearchCV(
                pipeline, param_grid, cv=args.cv_folds,
                scoring='f1_macro', n_jobs=args.n_jobs, verbose=1
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            print(f"最佳参数: {search.best_params_}")
            print(f"最佳交叉验证 F1-macro: {search.best_score_:.4f}")
    else:
        # 使用默认参数训练
        print("使用默认参数训练模型（跳过超参数搜索）...")
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline

    # 6. 测试集评估
    y_pred = best_pipeline.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['恒星', '星系']))
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))

    # 7. 保存分类错误的样本路径
    misclassified_mask = (y_test != y_pred)
    misclassified_paths = paths_test[misclassified_mask]
    n_mis = len(misclassified_paths)
    print(f"\n分类错误样本数: {n_mis}")
    if n_mis > 0:
        with open('misclassified_paths.txt', 'w') as f:
            for p in misclassified_paths:
                f.write(p + '\n')
        print("分类错误样本路径已保存到 misclassified_paths.txt")
    else:
        print("没有分类错误样本！")

    # 8. 保存模型
    save_data = {
        'pipeline': best_pipeline,
        'feature_config': {},
        'n_jobs': args.n_jobs,
        'search_used': args.search,
    }
    joblib.dump(save_data, args.output_model)
    print(f"模型已保存为 {args.output_model}")


if __name__ == "__main__":
    main()
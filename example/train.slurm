#!/bin/bash
#SBATCH --job-name=train_classifier
#SBATCH --partition=amd_m9_768
#SBATCH --nodes=1
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=64        
#SBATCH --mem=200G
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

ulimit -n 65536

source ~/.bashrc
conda activate sls_spec
cd /publicfs10/fs10-m9/home/m9s003101/sls_1d_spec/example

# 设置环境变量，防止 OpenMP 抢占核心
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 此时 os.cpu_count() 会正确返回 64，--n-jobs -1 将使用全部 64 个核
#生成光谱
python generate_challenge_spectra.py --cores 64 --large-re 5.0 --large-n 4.0

#提取拓扑特征
python extract_features.py --features betti entropy landscape amplitude --betti-bins 20 --landscape-layers 3 --landscape-bins 20 --amplitude-metric wasserstein --cores 64


#训练分类器
python train_classifier.py --save-test-paths --smote-ratio 0.3 --n-jobs -1




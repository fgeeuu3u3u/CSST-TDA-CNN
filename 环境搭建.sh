#!/bin/bash


# 创建 conda 环境
echo "创建 conda 环境 sls_spec ..."
conda create -n sls_spec python=3.10 -y

conda activate sls_spec



echo "安装依赖..."

conda install numpy==1.26.4 -y
conda install cython==3.0.6 -y
conda install mpi4py==3.1.6 -y
pip install -e . --no-build-isolation
pip install tqdm
pip install giotto-tda
pip install imbalanced-learn==0.12.3



echo "自动化安装完成！"


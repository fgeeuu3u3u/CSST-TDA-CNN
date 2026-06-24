<<<<<<< HEAD
# SLS_1D_SPEC

#### 介绍
一维无缝光谱仿真，包含望远镜的特性
1-d csst slitless spectroscopic simulation, include the charactor of optical system, grating, ccd and so on.

1. sh install.sh
2. cd example, and  "python sim_demo1.py"

If you use this software, you should include the following acknowledgments:
“This work is based on the scientific data processing software
system of the Chinese Space Station Telescope under the China
Manned Space Project(https://csst-tb.bao.ac.cn/code/zhangxin/sls_1d_spec)”

=======
<div align="center">
  <img src="https://github.com/fgeeuu3u3u/CSST-TDA-CNN/blob/ef8794c21634d5f5e54269824ace576b016eae7e/architecture.png" width="600" alt="项目架构图">
</div>

# CSST-TDA-CNN

在这里我们使用拓扑数据分析（TDA）以及卷积神经网络（CNN）对 CSST 中的二维光谱 + 多波段图像进行分类。

## 主要解决两个问题：

1. **低分辨率 + 高噪声**：使用 TDA 提取图像的整体结构特征（持久图 → 持续图像），获得对噪声鲁棒的拓扑描述符。
2. **类别严重不平衡**：在拓扑特征向量上应用 SMOTE（合成少数类过采样）生成少数类（如恒星）的合成样本，使训练集类别均衡，提升分类器对稀少天体的识别能力。
>>>>>>> 42ce73f136aed681301de2806d17f0ad786ad1c5

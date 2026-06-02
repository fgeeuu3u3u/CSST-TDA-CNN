# CSST-TDA-CNN
在这里我们使用拓扑数据分析TDA以及卷积神经网络对CSST中的二维光谱+多波段图像进行分类
主要解决两个问题：
1.低分辨率光谱包含较高的噪声，这里使用TDA来提取图像的整体特征
2.不同天体的数量差异大，这里使用SMOTE（Synthetic Minority Over-sampling TEchnique）来解决

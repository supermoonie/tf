[TOC]

# TensoFlow 实战

## 1、TensorFlow基础

1. 核心概念

   TensorFlow 中的计算可以表示为一个有向图（directed  graph），或者计算图（computation graph），其中每一个运算操作（operation）将作为一个节点（node），节点与节点之间的连接称为边（edge）。

   在计算图的边中流动（flow）的数据被称为张量（tensor）。

   Session 是用户使用TensorFlow 时的交互式接口。可通过Sesssion 的 Extend 方法添加新的节点和边，用以创建计算图，然后通过 Session.run() 执行计算图。

2. 实现原理

## 2、TensorFlow 第一步

#### TensorFlow 实现 Softmax Regression 识别手写数字

特征公式（i 代表第 i 类， j 代表一张图片的第 j 个像素）：
$$
feature_i = \sum_{j}{W_{i,j} x_j + b_i}
$$
对所有特征计算 softmax，就是都计算一个 exp 函数，然后再进行标准化（让所有类别输出的概率值和为 1）特征值越大的类，最后输出的概率也越大；反之，特征值越小的类，输出的概率也越小：
$$
softmax(x) = normalize(exp(x)) \\[4ex]
softmax(x) = \frac{exp(x_i)}{\sum_{j}{exp(x_j)}}
$$
![in_action_0](./images/in_action_0.png)

![in_action_1](./images/in_action_1.png)

![in_action_2](./images/in_action_2.png)

Cross-entropy 的定义（y 是预测的概率分布， $y^{\prime}$ 是真实的概率分布（即 Label 的 one-hot 编码））：
$$
H_{y^{\prime}(y) = - \sum_{i}{y_i^{\prime} \log(y_i)}}
$$
没有隐藏层的神经网络的整个流程：

- 定义算法公式，也就是神经网络 forward 时的计算
- 定义 loss，选定优化器，并指定优化器优化 loss
- 迭代地对数据进行训练
- 在测试集或验证集上对准确率进行评测

## 3、TensorFlow 实现自编码器及多层感知机

自编码器（AutoEncoder）：即可以使用自身的高阶特征编码自己。自编码器其实也是一种神经网络，它的输入和输出是一致的，它借助稀疏编码（Sparse Coding）的思想，目标是使用稀疏的一些高阶特征重新组合来重构自己。

自编码器：

- 去噪自编码器：去噪声，并保证隐含层节点小于输入层节点
- Masking Noise自编码器：将高斯噪声改为随机遮挡噪声
- Variational AutoEncoder（VAE）：VAE 对中间点的分布有强假设，拥有额外的损失项，且会使用特殊的 SGVB（Stochastic Gradient Variational Bayes）算法进行训练

Xavier 初始化器会根据某一层网络的输入、输出节点数量自动调整最合适的分布。如果深度学习模型的权重初始化得太小，信号将在每层间传递时主键缩小而难以产生作用，如果权重初始化得太大，信号将在每层间传递时逐渐放大并导致发散和失效。而Xavier 就是让权重满足 0 均值，同时方差为 $\frac{2}{n_{in} + n_{out}}$ 。




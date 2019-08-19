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
[./images/in_action_0.png](./images/in_action_0.png)






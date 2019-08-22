[TOC]

# TensorFlow 实战 Google 深度学习框架

## 1、TensorFlow 入门

### 1.1 TensorFlow 计算模型-计算图

Tensor - 张量

Flow - 流动，计算模型

在 TensorFlow 中，系统会自动维护一个默认的计算图

```python
# 默认的计算图
graph = tf.get_default_graph()
print(graph)
# 新的计算图
g_1 = tf.Graph()
    with g_1.as_default():
        v = tf.constant(1.0)
    with tf.Session(graph=g_1) as sess:
        v_value = sess.run(v)
        print(v_value)
        print(tf.get_default_graph())
```

​	在一个计算图中，可以通过集合（collection）来管理不通类别的资源。例如通过 tf.add_to_collection 可以将资源加入一个或多个集合中，然后通过 tf.get_collection 过去一个集合里面所有的资源。

TensorFlow 也自动管理了一些集合：

| 集合名称                              | 集合内容                         | 使用场景                 |
| ------------------------------------- | -------------------------------- | ------------------------ |
| tf.GraphKeys.VARIABLES                | 所有变量                         | 持久化模型               |
| tf.GraphKeys.TRAINABLE_VARIABLES      | 可学习的变量（神经网络中的参数） | 模型训练，生产可视化内容 |
| tf.GraphKeys.SUMMARIES                | 日志生成相关的张量               | 计算可视化               |
| tf.GraphKeys.QUEUE_RUNNERS            | 处理输入的QueueRunner            | 输入处理                 |
| tf.GraphKeys.MOVING_AVERAGE_VARIABLES | 所有计算了滑动平均值的变量       | 计算变量的滑动平均值     |

### 1.2 TensorFlow 数据模型-张量

​	**张量（tensor）** ，在TensorFlow 中的实现并不是直接采用数组的形式，它只是对TensorFlow 中运算结果的引用。在张量中并没有真正保存数字，它保存的是如何得到这些数字的计算过程。

张量的三个属性：名字（name）、维度（shape）和 类型（type）

张量的名字不仅是张量的唯一标识符，同样也给出了张量是如何计算的。

TensorFlow 支持14 中类型：实数（tf.float32、tf.float64）、整数（tf.int8、tf.int16、tf.int32、tf.int64、tf.uint8）和复数（tf.complex64、tf.complex128）

张量主要用于：

- 对中间计算结果的引用
- 当计算图构造完成之后，张量可以用来获得计算结果

### 1.3 TensorFlow 运行模型-会话

​	**会话（Session）** ，拥有并管理 TensorFlow 程序运行时的所有资源，所有计算完成后需要关闭会话以回收资源。

TensorFlow 提供了一种在交互式环境下直接构建默认会话的函数 tf.InteractiveSession ，这个函数会自动将生成的会话注册为默认会话。

tf.ConfigProto 用来配置生成的会话，通过 tf.ConfigProto 可以配置类似并行的线程数、GPU 分配策略、运算超时时间等参数。

tf.ConfigProto 常用参数：

- allow_soft_placement，为 True 时，在以下任一条件成立时，GPU 上的计算可以放到CPU 上运行：
  - 运算无法在GPU 上执行
  - 没有GPU 资源
  - 运算输入包含对 CPU 计算结果的应用
- log_device_placement，为 True 时，日志记录将会记录每个节点被安排在哪个设备上以方便调试

### 1.4 TensorFlow 实现神经网络

在 TensorFlow 中，变量（tf.Variable）的作用就是保存和更新神经网络中的参数。

TensorFlow 随机数生成器：

| 函数名              | 随机数分布                                                  | 主要参数                                |
| ------------------- | ----------------------------------------------------------- | --------------------------------------- |
| tf.random_normal    | 正态分布                                                    | 平均值、标准差、取值类型                |
| tf.truncated_normal | 正态分布，但如果随机出来的值超过 2 个标准差，将会被重新随机 | 平均值、标准差、取值类型                |
| tf.random_uniform   | 均匀分布                                                    | 最小、最大取值，取值类型                |
| tf.random_gamma     | Gamma 分布                                                  | 形状参数 alpha、尺度参数 beta、取值类型 |

TensorFlow 常数生成函数：

| 函数名      | 功能                         | 样例                                            |
| ----------- | ---------------------------- | ----------------------------------------------- |
| tf.zeros    | 产生全 0  的数组             | tf.zero([2, 3], int32)->[[0, 0, 0,], [0, 0, 0]] |
| tf.ones     | 产生全 1 的数组              | tf.ones([2, 3], int32)->[[1, 1, 1], [1, 1, 1]]  |
| tf.fill     | 产生一个全部为给定数字的数组 | tf.fill([2, 3],  9)->[[9, 9, 9], [9, 9, 9]]     |
| tf.constant | 产生一个给定值的常量         | tf.constant([1, 2, 3])->[1, 2, 3]               |

**监督学习的思想** 就是在已知的答案的标注数据集上，模型给出的预测要尽量接近真实的答案。通过调整神经网络中的参数对训练数据进行拟合，可以使得模型对未知的样本提供预测的能力。

在神经网络优化算法中，最常用的方法是反向传播算法（backpropagataion）：

![tf_in_action_01](./images/tf_in_action_01.png)

**损失函数：** 计算当前预测值与真实值之间的差距

## 2、深层神经网络

### 2.1 深度学习与深层神经网络

激活函数实现去线性化：

![tf_in_action_02](./images/tf_in_action_02.png)

常用的激活函数：

![tf_in_action_03](./images./tf_in_action_03.png)


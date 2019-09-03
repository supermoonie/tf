import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常数
INPUT_NODE = 784  # 输入层的节点数，对于MNIST 的数据集，这个等于图片的像素
OUTPUT_NODE = 10  # 输出层的节点数，这个等于类别的数目。因为在MNIST 数据集中需要区分的是 0-9 这10 个数字，所以输出层的节点数为 10

# 配置神经网络的参数
LAYER1_NODE = 500  # 隐藏层节点数，这里使用只有一个隐藏层的网络结构作为样例。这个隐藏层有 500 个节点。
BATCH_SIZE = 100  # 一个训练 batch 中的训练数据个数，数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降。
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 5000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这里
# 定义了一个使用 ReLU 激活函数的三层全连接神经网络。通过加入隐藏层实现了多层网络结构，
# 通过 ReLU 激活函数实现了去线性化。在这个激活函数中也支持传入用于计算参数平均值的类，
# 这样方便在测试时使用滑动平均模型。
def inference(input_tensor, avg_class, reuse=False):
    if avg_class is None:
        # 定义第一层神经网络的变量和前向传播过程
        with tf.variable_scope('layer1', reuse=reuse):
            # 根据传进来的 reuse 来判断是创建变量还是使用已经创建好的。
            # 在第一次构造网络时需要创建新的变量，以后每次调用这个函数都直接使用 reuse=True
            # 就不需要每次将变量传进来了
            weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [LAYER1_NODE],
                                     initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

        # 类似地定义第二层神经网络的变量和前向传播过程。
        with tf.variable_scope('layer2', reuse=reuse):
            weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [OUTPUT_NODE],
                                     initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, weights) + biases
        return layer2
    else:
        # 定义第一层神经网络的变量和前向传播过程
        with tf.variable_scope('layer1', reuse=reuse):
            # 根据传进来的 reuse 来判断是创建变量还是使用已经创建好的。
            # 在第一次构造网络时需要创建新的变量，以后每次调用这个函数都直接使用 reuse=True
            # 就不需要每次将变量传进来了
            weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [LAYER1_NODE],
                                     initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(
                tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases)
            )

        # 类似地定义第二层神经网络的变量和前向传播过程。
        with tf.variable_scope('layer2', reuse=reuse):
            weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [OUTPUT_NODE],
                                     initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases)
        return layer2


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None，
    # 所以函数不会使用参数的滑动平均值。
    y = inference(x, None)

    # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里指定这个变量为
    # 不可训练的变量。在使用 TensorFlow 训练神经网络时，
    # 一般会将代表训练轮数的变量指定为不可训练的参数。
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。其它辅助变量（比如 global_step）就
    # 不需要了。tf.trainable_variables 返回的就是图上集合
    # GraphKeys.TRAINABLE_VARIABLES 中的元素。这个集合的元素就是所有没有指定
    # trainable=False 的参数。
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果。滑动平均不会改变变量本身的取值，
    # 而是会维护一个影子变量来记录其滑动平均值。所以当需要使用这个滑动平均值时，
    # 需要明确调用 average 函数。
    average_y = inference(x, variable_averages, True)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用 TensorFlow 中
    # 提供的 sparse_softmax_cross_entropy_with_logits 函数来计算交叉熵。当分类
    # 问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。MNIST 问题的图片中
    # 只包含了 0-9 中的一个数字，所以可以使用这个函数来计算交叉熵损失。
    # 这个函数的第一个参数是神经网络不包括 softmax 层的前向传播结果，
    # 第二个是训练数据的正确答案。因为标准答案是一个长度为 10 的一维数组，
    # 而该函数需要提供的是一个正确答案的数字，所以需要使用 tf.argmax 函数来得到正确答案对应的类别编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前 batch 中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算 L2 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项。
    with tf.variable_scope('', reuse=True):
        regularization = regularizer(tf.get_variable('layer1/weights', [INPUT_NODE, LAYER1_NODE])) + \
                         regularizer(tf.get_variable('layer2/weights', [LAYER1_NODE, OUTPUT_NODE]))
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY)  # 学习率衰减速度

    # 使用 tf.train.GradientDescentOptimizer 优化算法来优化损失函数。注意这里
    # 损失函数包含了交叉熵损失和 L2 正则化损失。
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，
    # 又需要更新每一个参数的滑动平均值。为了一次完成多个操作，TensorFlow 提供了
    # tf.control_dependencies 和 tf.group 两种机制。下面两行程序和
    # train_op = tf.group(train_step, variable_averages_op) 是等价的。
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向传播网络结果是否正确。tf.argmax(average_y, 1)
    # 计算每一个样例的预测答案。其中 average_y 是一个 batch_size * 10 的二维数组，每一行
    # 表示一个样例的前向传播结果。tf.argmax 的第二个参数表示选取最大值的操作仅在第一个维度中进行，
    # 也就是说，只在每一行选取最大值对应的下标。于是得到的结果是一个长度为 batch 的一维数组，
    # 这个一维数组中的值就表示了每一个样例对应的数字识别结果。tf.equal 判断两个张量的每一维是否相等，如果相等返回True，否则返回False。
    current_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 这个运算首先将一个布尔型的数值转换为实数型，然后计算平均。这个平均值就是
    # 模型在这一组数据上的正确率。
    accuracy = tf.reduce_mean(tf.cast(current_prediction, tf.float32))

    # 初始化会话并开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的
        # 条件和判断训练的效果。
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 准备测试数据。在真实的应用中，这部分数据在训练时是不可见的，这个数据
        # 只是作为模型优劣的最后评价标准。
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        # 迭代训练神经网络
        for i in range(TRAINING_STEPS):
            # 每 100 轮输出一次在验证数据集上的测试结果。
            if i % 100 == 0:
                # 计算滑动平均模型在验证数据上的结果。因为MNIST 数据集比较小，所以一次
                # 可以处理所有的验证数据。为了计算方便，没有将验证数据划分为更小的batch。
                # 当神经网络模型比较复杂或者验证数据比较大时，太大的batch
                # 会导致计算时间过长甚至发生内存溢出的错误。
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('after %d training step(s), validation accuracy '
                      'using average model is %g ' % (i, validate_acc))

            # 产生这一轮使用的一个 batch 的训练数据，并运行训练过程。
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束之后，在测试数据上检测神经网络网络模型的最终正确率。
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('after %d training step(s), test accuracy using average model is %g ' % (TRAINING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets('../mnist/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()

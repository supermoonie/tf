from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('../mnist/', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# None 代表不限条数的输入
x = tf.placeholder(tf.float32, [None, 784])
# 10 代表有 10 类
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
init = tf.global_variables_initializer()
# 为了训练模型，需要定义一个 loss function 来描述模型对问题的分类精度
# loss 越小，代表模型的分类结果与真实值的偏差越小，即该模型越精确
# 训练的目的就是不断将这个 loss 减小，知道达到一个全局最优或者局部最优解
# 对于多分类问题，通常使用 cross-entropy 作为 loss function
y_ = tf.placeholder(tf.float32, [None, 10])  # 真实的label
# tf.reduce_sum 就是求和，tf.reduce_mean 用来对每个 batch 数据结果求平均值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 随机梯度下降SGD
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        # 每次随机从训练集中抽取 100 条样本构成一个 mini-batch，并 feed 给 placeholder
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, loss, y_result = sess.run([train_step, cross_entropy, y_], feed_dict={x: batch_xs, y_: batch_ys})
        # tf.argmax 是从一个 tensor 中寻找最大值的序号，
        # tf.argmax(y, 1) 就是求各个预测的数字中概率最大的那一个
        # tf.argmax(y_, 1) 则是找样本的真实数字类别
        # tf.equal 方法则用来判断预测的数字类别是否是正确的类别，最后返回计算分类是否正确的操作
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 统计全部样本预测的 accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('index: {}, accuracy: {}, loss: {}'.format(i, accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}), loss))


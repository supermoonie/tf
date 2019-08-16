from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


tf.app.flags.DEFINE_integer('is_train', 1, '指定是否训练模型，还是拿数据预测')
FLAGS = tf.app.flags.FLAGS


def full_connection():
    """
    用全连接来对手写数字进行识别
    :return:
    """
    # 1. 准备数据
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # 2. 构建模型
    weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10]))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10]))
    y_predict = tf.matmul(x, weights) + bias

    # 3. 构造损失函数
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4. 优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 5. 准确率计算
    equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init = tf.global_variables_initializer()

    # saver = tf.train.Saver()

    with tf.Session() as sess:
        if FLAGS.is_train == 1:
                sess.run(init)
                image, label = mnist.train.next_batch(100)
                print('训练之前， 损失值: %f' % sess.run(error, feed_dict={x: image, y_true: label}))
                for i in range(3000):
                    _, loss, accuracy_value = sess.run([optimizer, error, accuracy], feed_dict={x: image, y_true: label})
                    print('第 %d 次训练，损失值为: %f, 准确率为: %f, ' % (i + 1, loss, accuracy_value))
        else:
            pass
            # for i in range(100):
            #     mnist_x, mnist_y = mnist.test.next_batch(1)
            #     print('第 %d 个样本的真实值：%d，模型预测结果为：%d' % (i + 1, tf.argmax(sess.run(y_true, feed_dict={x: mnist_x, y_true: mnist_y}), )))

    return None


if __name__ == '__main__':
    full_connection()

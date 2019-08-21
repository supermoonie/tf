from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def cnn():
    mnist = input_data.read_data_sets('../mnist/', one_hot=True)
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        # 卷积层 1
        W_conv1 = weigth_variable([5, 5, 1, 32], name='W_conv1')
        b_conv1 = bias_variable([32], name='b_conv1')
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        # 卷积层 2
        W_conv2 = weigth_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        # 全连接层 1
        W_fc1 = weigth_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        keep_prod = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prod)
        # 全连接层 2
        W_fc2 = weigth_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='softmax')
        # 损失函数
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # 准确率计算
        current_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(current_prediction, tf.float32))

        tf.global_variables_initializer().run()
        for i in range(500):
            batch = mnist.train.next_batch(50)
            if i % 10 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prod: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prod: 0.5})
        print('test accuracy %g' % (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels,
                                                                  keep_prod: 1.0})))


def weigth_variable(shape, name=None):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    cnn()

import tensorflow as tf
from numpy.random import RandomState


def demo01():
    a = tf.constant([1.0, 2.0], name='a')
    b = tf.constant([2.0, 3.0], name='b')
    result = a + b
    with tf.Session() as sess:
        result_value = sess.run(result)
        print(result_value)


def graph_demo():
    graph = tf.get_default_graph()
    print(graph)
    g_1 = tf.Graph()
    with g_1.as_default():
        v = tf.constant(1.0)
    with tf.Session(graph=g_1) as sess:
        v_value = sess.run(v)
        print(v_value)
        print(tf.get_default_graph())
    g_2 = tf.Graph()
    with g_2.as_default():
        v = tf.constant(2.0)
    with tf.Session(graph=g_2) as sess:
        v_value = sess.run(v)
        print(v_value)
        print(tf.get_default_graph())


def config_demo():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(config=config):
        pass


def variable_demo():
    batch_size = 8

    w_1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
    w_2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

    x = tf.placeholder(tf.float32, shape=(None, 2), name='input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='out')

    a = tf.matmul(x, w_1)
    y = tf.matmul(a, w_2)
    y = tf.sigmoid(y)
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                    + (1-y) * tf.log(tf.clip_by_value(1-y, 12-10, 1.0)))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    rdm = RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size, 2)
    Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(500):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
            if i % 10 == 0:
                total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
                print('after %d training step, cross entropy on all data is %g' %(i, total_cross_entropy))
        print(sess.run(w_1))
        print(sess.run(w_2))


if __name__ == '__main__':
    # demo01()
    # graph_demo()
    # config_demo()
    variable_demo()

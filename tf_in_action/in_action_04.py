import tensorflow as tf


def cross_entropy():
    cross = -tf.reduce_mean([1, 0, 0] * tf.log(tf.clip_by_value([0.2, 0.7, 0.1], 1e-10, 1.0)))
    with tf.Session() as sess:
        cross_value = sess.run(cross)
        print(cross_value)


def tf_clip_by_value():
    v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    clip = tf.clip_by_value(v, 2.5, 4.5)
    with tf.Session() as sess:
        v_v = sess.run(clip)
        print(v_v)


def tf_softmax_cross_entropy_with_logits():
    cross = tf.nn.softmax_cross_entropy_with_logits_v2(labels=[1, 0, 0], logits=[0.2, 0.7, 0.1])
    with tf.Session() as sess:
        cross_value = sess.run(cross)
        print(cross_value)


def tf_exponential_decay():
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.1, global_step=global_step, decay_steps=100, decay_rate=0.96, staircase=True)
    cross = -tf.reduce_mean([1, 0, 0] * tf.log(tf.clip_by_value([0.2, 0.7, 0.1], 1e-10, 1.0)))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
        .minimize(loss=cross, global_step=global_step)


def tf_regularizer():
    w = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
    with tf.Session() as sess:
        # 输出为(|1| + |-2| + |-3| + |4|)*0.5 = 5
        print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(w)))
        # 输出为(1^2 + (-2)^2 + (-3)^2 + 4^2)/2*0.5 = 7.5
        print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(w)))


def tf_exponential_moving_average():
    # 定义一个变量用于计算滑动平均，这个变量的初始值为 0。注意这里手动指定了变量的
    # 类型为 tf.float32，因为所有需要计算滑动平均的变量必须是实数型。
    v_1 = tf.Variable(0, dtype=tf.float32)
    # 这里num_updates 变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率
    num_updates = tf.Variable(0, trainable=False)

    # 定义一个滑动平均的类（class）。初始化时给定了衰减率（0.99）和控制衰减率的变量 num_updates
    ema = tf.train.ExponentialMovingAverage(decay=0.99, num_updates=num_updates)
    # 定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个操作时
    # 这个列表中的变量都会被更新
    maintain_averages_op = ema.apply([v_1])

    with tf.Session() as sess:
        # 初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 通过 ema.average(v_1) 获取滑动平均之后变量的取值。在初始化之后变量 v_1 的值和 v_1 的
        # 滑动平均都为 0
        print(sess.run([v_1, ema.average(v_1)]))    # 输出 [0.0. 0.0]

        # 更新变量 v_1 的值到 5
        sess.run(tf.assign(v_1, 5))
        # 更新 v_1 的滑动平均值，衰减率为 main{0.99, (1+num_updates)/(10+num_updates) = 0.1} = 0.1
        # 所以 v_1 的滑动平均会被更新为 0.1*0 + 0.9*5 = 4.5
        sess.run(maintain_averages_op)
        print(sess.run([v_1, ema.average(v_1)]))    # 输出 [5.0, 4.5]

        # 更新 num_updates 的值为 10000
        sess.run(tf.assign(num_updates, 10000))
        # 更新 v_1 的值为 10
        sess.run(tf.assign(v_1, 10))
        # 更新 v_1 的滑动平均值，衰减率为 min{0.99, (1+num_updates)/(10+num_updates) = 0.999} = 0.99
        # 所以 v_1 的滑动平均会被更新为 0.99 * 4.5 + 0.01 * 10 = 4.555
        sess.run(maintain_averages_op)
        print(sess.run([v_1, ema.average(v_1)]))    # 输出 [10.0, 4.5549998]

        # 再次更新滑动平均值，得到的新滑动平均值为 0.99 * 4.555 + 0.01 * 10 = 4.60945
        sess.run(maintain_averages_op)
        print(sess.run([v_1, ema.average(v_1)]))    # 输出 [10.0, 4.6094499]


if __name__ == '__main__':
    # cross_entropy()
    # tf_clip_by_value()
    # tf_softmax_cross_entropy_with_logits()
    # tf_regularizer()
    tf_exponential_moving_average()

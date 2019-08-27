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


if __name__ == '__main__':
    # cross_entropy()
    # tf_clip_by_value()
    # tf_softmax_cross_entropy_with_logits()
    tf_regularizer()

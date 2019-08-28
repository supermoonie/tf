from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('../mnist/', one_hot=True)


def data_info():
    print('training data size: ', mnist.train.num_examples)
    print('validating data size: ', mnist.validation.num_examples)
    print('testing data size: ', mnist.test.num_examples)
    print('example training data: ', mnist.train.images[0])
    print('example training data label: ', mnist.train.labels[0])
    batch_size = 100
    xs, ys = mnist.train.next_batch(batch_size=batch_size)
    print('X shape: ', xs.shape)
    print('Y shape: ', ys.shape)


def tf_get_variable():
    # 在名称为 foo 的命名空间内创建名字为 v 的变量
    with tf.variable_scope('foo'):
        v = tf.get_variable('v', [1], initializer=tf.constant_initializer(1.0))

    # 因为在命名空间 foo 中已经存在名字为 v 的变量，所以以下代码将报错:
    # Variable foo/v already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
    # with tf.variable_scope('foo'):
    #     v = tf.get_variable('v', [1])

    # 在生成上下文管理器时，将参数 reuse 设置为 True。
    # 这样 tf.get_variable 函数将直接获取已经声明得变量。
    with tf.variable_scope('foo', reuse=True):
        v1 = tf.get_variable('v', [1])
        print(v == v1)

    # 将参数 reuse 设置为 True 时，tf.variable_scope 将只能获取已经创建过的变量。
    # 因为在命名空间 bar 中还没有创建变量 v ，所以以下代码将报错:
    # Variable bar/v does not exist, or was not created with tf.get_variable().
    # Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
    with tf.variable_scope('bar', reuse=True):
        v = tf.get_variable('v', [1])


if __name__ == '__main__':
    # data_info()
    tf_get_variable()

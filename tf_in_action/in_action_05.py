from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import  gfile


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


def tf_saver():
    v_1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v_1')
    v_2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v_2')
    result = v_1 + v_2

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.save(sess, './model/model.ckpt')


def tf_model_restore():
    v_1 = tf.Variable(tf.constant(1.0, shape=[1]), name='other-v1')
    v_2 = tf.Variable(tf.constant(2.0, shape=[1]), name='other-v2')
    result = v_1 + v_2

    # 使用一个字典来重命名变量就可以加载原来的模型。
    # 这个字典制定了原来名称为 v_1 的变量现在加载到 v_1 中（名称为 other-v1），
    # 名称为 v_2 的变量加载到变量 v_2 中（名称为 other-v2）
    saver = tf.train.Saver({'v_1': v_1, 'v_2': v_2})
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, './model/model.ckpt')
        print(sess.run(result))


def tf_saver_moving_average():
    v = tf.Variable(0, dtype=tf.float32, name='v')
    # 在没有申明滑动平均模型时只有一个变量 v， 所以以下语句只会输出 'v:0'。
    for variables in tf.global_variables():
        print(variables.name)

    ema = tf.train.ExponentialMovingAverage(0.99)
    maintain_averages_op = ema.apply(tf.global_variables())
    # 在申明滑动平均模型之后，TensorFlow 会自动生成一个影子变量 v/ExponentialMoving Average。
    # 于是以下语句会输出 'v:0' 和 'v/ExponentialMovingAverage:0'
    for variables in tf.global_variables():
        print(variables.name)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        sess.run(tf.assign(v, 10))
        sess.run(maintain_averages_op)
        # 保存时，TensorFlow 将 v:0 和 v/ExponentialMovingAverage:0 两个变量都存下来。
        saver.save(sess, './model/model_2.ckpt')
        print(sess.run([v, ema.average(v)]))


def tf_saver_restore_moving_average():
    v = tf.Variable(0, dtype=tf.float32, name='v')
    ema = tf.train.ExponentialMovingAverage(0.99)
    # 通过使用 variables_to_restore 函数方便加载时重命名滑动平均变量
    # {'v/ExponentialMovingAverage': v}
    # 以下代码会输出：{'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
    # 其中后面的 Variable 类就代表了变量 v。
    print(ema.variables_to_restore())

    # 通过变量重命名将原来变量 v 的滑动平均值直接赋值给 v。
    # saver = tf.train.Saver({'v/ExponentialMovingAverage': v})
    saver = tf.train.Saver(ema.variables_to_restore())
    with tf.Session() as sess:
        saver.restore(sess, './model/model_2.ckpt')
        print(sess.run(v))


def tf_save_model_to_pb():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 导出当前计算图的 GraphDef 部分，只需要这一部分就可以完成从输入层到输出层的计算过程。
        graph_def = tf.get_default_graph().as_graph_def()

        # 将图中的变量及其取值转化为常量，同时将图中不必要的节点去掉。
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
        # 将导出的模型存入文件
        with tf.gfile.GFile('./model/model_3.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())


def tf_save_model_to_json():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.export_meta_graph('./model/model_4.ckpt.meda.json', as_text=True)


def tf_restore_from_pb():
    with tf.Session() as sess:
        # 读取保存的模型文件，并将文件解析成对应的 GraphDef Protocol Buffer
        with gfile.FastGFile('./model/model_3.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # 将 graph_def 中保存的图加载到当前的图中。return_elements=['add:0'] 给出了返回的张量的名称。
        # 在保存的时候给出的是计算节点的名称，所以为 'add'。在加载的时候给出的是张量的名称，所以是 'add:0'。
        result = tf.import_graph_def(graph_def, return_elements=['add:0'])
        print(sess.run(result))


if __name__ == '__main__':
    # data_info()
    # tf_get_variable()
    # tf_saver()
    # tf_model_restore()
    # tf_saver_moving_average()
    # tf_saver_restore_moving_average()
    # tf_save_model_to_pb()
    # tf_restore_from_pb()
    tf_save_model_to_json()

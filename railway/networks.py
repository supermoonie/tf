import tensorflow as tf
import numpy as np


class CNN(object):
    def __init__(self, img_height, img_width, class_set, model_save_dir):
        self.img_height = img_height
        self.img_width = img_width
        self.model_save_dir = model_save_dir
        self.class_set = class_set
        self.w_alpha = 0.01
        self.b_alpha = 0.1
        self.X = tf.placeholder(tf.float32, [None, self.img_height * self.img_width], name='X')
        self.Y_ = tf.placeholder(tf.float32, [None, len(self.class_set)], name='Y_')
        self.keep_prob = tf.placeholder(tf.float32)

    @staticmethod
    def convert2gray(img):
        """
        图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
        :param img:
        :return:
        """
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    def text2vec(self, text):
        """
        转标签为oneHot编码
        :param text: str
        :return: numpy.array
        """
        vector = np.zeros(len(self.class_set))
        idx = self.class_set.index(text)
        vector[idx] = 1
        return vector

    def model(self):
        x = tf.reshape(self.X, shape=[-1, self.img_height, self.img_width, 1])

        # 卷积层1
        w_1 = tf.get_variable(name='w_1', shape=[3, 3, 1, 32], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        b_1 = tf.Variable(self.b_alpha * tf.random_normal([32]))
        h_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_1, strides=[1, 1, 1, 1], padding='SAME'), b_1))
        h_1 = tf.nn.max_pool(h_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_1 = tf.nn.dropout(h_1, self.keep_prob)

        # 卷积层2
        w_2 = tf.get_variable(name='w_2', shape=[3, 3, 32, 64], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        b_2 = tf.Variable(self.b_alpha * tf.random_normal([64]))
        h_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_1, w_2, strides=[1, 1, 1, 1], padding='SAME'), b_2))
        h_2 = tf.nn.max_pool(h_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_2 = tf.nn.dropout(h_2, self.keep_prob)

        # 卷积层3
        w_3 = tf.get_variable(name='w_3', shape=[3, 3, 64, 128], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        b_3 = tf.Variable(self.b_alpha * tf.random_normal([128]))
        h_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_2, w_3, strides=[1, 1, 1, 1], padding='SAME'), b_3))
        h_3 = tf.nn.max_pool(h_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h_3 = tf.nn.dropout(h_3, self.keep_prob)
        print(">>> convolution 3: ", h_3.shape)
        next_shape = h_3.shape[1] * h_3.shape[2] * h_3.shape[3]
        print(">>> next_shape: ", next_shape)

        # 全连接层1
        w_4 = tf.get_variable(name='w_4', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        b_4 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(h_3, [-1, w_4.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_4), b_4))
        dense = tf.nn.dropout(dense, self.keep_prob)

        # 全连接层2
        w_out = tf.get_variable('w_out', shape=[1024, len(self.class_set)], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        b_out = tf.Variable(self.b_alpha * tf.random_normal([len(self.class_set)]))

        y_predict = tf.add(tf.matmul(dense, w_out), b_out)

        return y_predict

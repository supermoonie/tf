import tensorflow as tf
import numpy as np


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

        # 卷积层 1
        W_conv1 = weigth_variable(shape=[5, 5, 1, 32], name='W_conv1')
        b_conv1 = bias_variable([32], name='b_conv1')
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        print('h_pool1.shape: ', h_pool1.shape)
        # 卷积层 2
        W_conv2 = weigth_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        # 全连接层 1
        W_fc1 = weigth_variable([17 * 17 * 64, 1024])
        b_fc1 = bias_variable([1024])
        print('h_pool2.shape: ', h_pool2.shape)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 17 * 17 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob)
        # 全连接层 2
        W_fc2 = weigth_variable([1024, len(self.class_set)])
        b_fc2 = bias_variable([len(self.class_set)])
        y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='softmax')

        return y_predict

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sklearn.preprocessing as prep


# Xavier 初始化器会根据某一层网络的输入、输出节点数量自动调整最合适的分布。
# 如果深度学习模型的权重初始化得太小，信号将在每层间传递时主键缩小而难以产生作用
# 如果权重初始化得太大，信号将在每层间传递时逐渐放大并导致发散和失效
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']),
                                           self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        self.loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def partial_fit(self, X):
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return loss

    def calc_total_loss(self, X):
        return self.sess.run(self.loss, feed_dict={self.x: X, self.scale: self.training_scale})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_biases(self):
        return self.sess.run(self.weights['b1'])

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def multi_layer_perceptron():
    mnist = input_data.read_data_sets('../mnist/', one_hot=True)
    with tf.Session() as sess:
        in_units = 784
        h1_units = 300
        W1 = tf.Variable(initial_value=tf.truncated_normal([in_units, h1_units], stddev=0.1))
        b1 = tf.Variable(tf.zeros([h1_units]))
        W2 = tf.Variable(tf.zeros([h1_units, 10]))
        b2 = tf.Variable(tf.zeros([10]))

        x = tf.placeholder(tf.float32, [None, in_units])
        keep_prob = tf.placeholder(tf.float32)

        hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
        hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
        y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(500):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, loss = sess.run([train_step, cross_entropy], {x: batch_xs, y_: batch_ys, keep_prob: 0.75})
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(
                'index: {}, accuracy: {}, loss: {}'.format(i, accuracy.eval(
                    {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}),
                                                           loss))


if __name__ == '__main__':
    mnist = input_data.read_data_sets('../mnist/', one_hot=True)
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1

    auto_encoder = AdditiveGaussianNoiseAutoEncoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                                    optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = auto_encoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
        if epoch % display_step == 0:
            print('epoch: {}, avg_cost: {}'.format((epoch + 1), avg_cost))
    print('total cost: ' + str(auto_encoder.calc_total_loss(X_test)))


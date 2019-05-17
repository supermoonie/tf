import tensorflow as tf
import numpy as np
import cv2 as cv


def get_x(a):
    image = np.zeros([28, 28], dtype=np.uint8)
    cv.putText(image, str(a), (7, 21), cv.FONT_HERSHEY_PLAIN, 1.3, (255), 2, 8)
    cv.imshow('image', image)
    data = np.reshape(image, [1, 784])
    return data / 255


def feed_fetch():
    x = tf.placeholder(shape=[1, 784], dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)
    w = tf.Variable(tf.random_normal([784, 1]))
    b = tf.Variable(tf.random_normal([1, 1]))

    y_ = tf.add(tf.matmul(x, w), b)
    loss = tf.reduce_sum(tf.square(tf.subtract(y, y_)))
    train = tf.train.GradientDescentOptimizer(0.01)
    step = train.minimize(loss)
    init = tf.global_variables_initializer()
    x_input = get_x(4)
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            y_result, curr_loss, curr_step = sess.run([y_, loss, step], feed_dict={x: x_input, y: 4})
            print('y_ : %f, loss : %f' % (y_result, curr_loss))
        curr_w, curr_b = sess.run([w, b], feed_dict={x: x_input, y: 4})
        print('curr_w : ', curr_w)
        print('curr_b : ', curr_b)


if __name__ == '__main__':
    # get_x(4)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    feed_fetch()

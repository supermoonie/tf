import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from PIL import Image

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from railway.networks import CNN

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class TrainModel(CNN):
    def __init__(self, img_path, cycle_stop, right_rate_stop, cycle_save=500):
        self.img_path = img_path
        self.cycle_stop = cycle_stop
        self.right_rate_stop = right_rate_stop
        self.cycle_save = cycle_save
        self.class_set = [f for f in os.listdir(img_path)]
        self.img_list = self.collection_all_img()
        self.train_img_set, self.test_img_set = self.train_test_split(test_size=0.2)
        self.img_width = 66
        self.img_height = 66
        super(TrainModel, self).__init__(self.img_width, self.img_height, self.class_set, './model/')

    def collection_all_img(self):
        """
        收集训练的图片
        :return:
        """
        c_path = [(f, os.path.join(self.img_path, f)) for f in os.listdir(self.img_path)]
        image_path_set = []
        for label, path in c_path:
            image_path_set.extend([(label, os.path.join(path, f)) for f in os.listdir(path)])
        # 打乱文件顺序
        random.seed(time.time())
        random.shuffle(image_path_set)
        return image_path_set

    def train_test_split(self, test_size=0.2):
        """
        按比例将图像集划分为训练集与测试集
        :param test_size:
        :return:
        """
        split_index = int(len(self.img_list) * test_size)
        return self.img_list[split_index:], self.img_list[0: split_index]

    def get_batch(self, img_set, size=128):
        """
        获取一批训练集
        :param img_set:
        :param size:
        :return:
        """
        batch_x = np.zeros([size, self.img_height * self.img_width])  # 初始化
        batch_y = np.zeros([size, len(self.class_set)])  # 初始化
        idx = np.random.randint(low=size, high=len(self.train_img_set) - size)
        index = 0
        for label, img_path in img_set[idx: idx + size]:
            captcha_image = Image.open(img_path)
            # gray_image = captcha_image.convert('L')
            captcha_array = np.array(captcha_image)
            image_array = self.convert2gray(captcha_array)
            # batch_x[index, :] = image_array.flatten() / 255
            batch_x[index, :] = image_array.flatten()
            batch_y[index, :] = self.text2vec(label)
            index = index + 1
        return batch_x, batch_y

    def train(self):
        y_predict = self.model()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict, labels=self.Y_))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        # 计算准确率
        current_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(self.Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(current_prediction, tf.float32))
        saver = tf.train.Saver()

        step = 1
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(self.cycle_stop):
                batch_x, batch_y = self.get_batch(self.train_img_set, size=128)
                _, cost_ = sess.run([optimizer, cost],
                                    feed_dict={self.X: batch_x, self.Y_: batch_y, self.keep_prob: 0.75})
                if step % 10 == 0:
                    batch_x_test, batch_y_test = self.get_batch(self.test_img_set, size=100)
                    test_right_rate = sess.run(accuracy,
                                               feed_dict={self.X: batch_x_test, self.Y_: batch_y_test,
                                                          self.keep_prob: 1.})
                    print("第{}次训练 >>> ".format(step))
                    print("[训练集] 准确率为 {:.5f} >>> loss {:.10f}".format(test_right_rate, cost_))

                    batch_x_verify, batch_y_verify = self.get_batch(self.train_img_set, size=100)
                    train_right_rate = sess.run(accuracy,
                                                feed_dict={self.X: batch_x_verify, self.Y_: batch_y_verify,
                                                           self.keep_prob: 1.})
                    print("[验证集] 准确率为 {:.5f} >>> loss {:.10f}".format(train_right_rate, cost_))
                    if train_right_rate > self.right_rate_stop and test_right_rate > self.right_rate_stop:
                        saver.save(sess, self.model_save_dir)
                        print("验证集准确率达到99%，保存模型成功")
                        break
                if i % self.cycle_save == 0:
                    saver.save(sess, self.model_save_dir)
                    print("定时保存模型成功")
                step += 1
            saver.save(sess, self.model_save_dir)


def main():
    train = TrainModel('D:/mac_temp/captcha/', 10000, 0.99, 500)
    # class_set = train.collection_all_img()
    # print(class_set)
    # print(len(class_set))
    # train_set, test_set = train.train_test_split()
    # print(train_set)
    # print(len(train_set))
    # print(test_set)
    # print(len(test_set))
    # batch_x, batch_y = train.get_batch(train.train_img_set, size=1)
    # plt.imshow(np.reshape(batch_x[0], [66, 66]))
    # print(batch_x[0])
    # print(train.class_set)
    # print(batch_y[0])
    # plt.show()
    train.train()


if __name__ == '__main__':
    main()

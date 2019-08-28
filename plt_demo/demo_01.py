import numpy as np
from matplotlib import pyplot as plt


def demo_01():
    x = np.arange(-11, 12)
    y = x * x
    plt.title('demo_01')
    plt.xlabel('x axis caption')
    plt.ylabel('y axis caption')
    plt.plot(x, y)
    plt.show()


def demo_02():
    x = np.arange(-11, 12)
    y = x * x
    plt.title('demo_02')
    plt.xlabel('x axis caption')
    plt.ylabel('y axis caption')
    plt.plot(x, y, 'or')
    plt.show()


def demo_03():
    x = np.arange(0, 3 * np.pi, 0.1)
    y = np.sin(x)
    plt.title('demo_03')
    plt.plot(x, y)
    plt.show()


def demo_04():
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    # 两行一列
    plt.subplot(2, 1, 1)
    plt.title('sin')
    plt.plot(x, y_sin)
    plt.subplot(2, 1, 2)
    plt.title('cos')
    plt.plot(x, y_cos)
    plt.show()


def demo_05():
    x_1 = [5, 8, 10]
    y_1 = [12, 16, 6]
    plt.bar(x_1, y_1, align='center')
    x_2 = [6, 9, 11]
    y_2 = [6, 15, 7]
    plt.bar(x_2, y_2, align='center', color='g')
    plt.title('bar_demo')
    plt.show()


def demo_06():
    a = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])
    plt.hist(a, bins=[0, 20, 40, 60, 80, 100])
    plt.title("histogram")
    plt.show()


if __name__ == '__main__':
    # demo_01()
    # demo_02()
    # demo_03()
    # demo_04()
    # demo_05()
    demo_06()

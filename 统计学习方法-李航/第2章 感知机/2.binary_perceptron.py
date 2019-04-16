import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from pltGif import pltGif

training_set = [[1, 1, 1], [2, 4, 1], [3, 1, -1], [4, 2, -1]]  # 训练数据集
w = [1, 1]  # 参数初始化
b = 0
history = []  # 用来记录每次更新过后的w,b
history.append([copy.copy(w), b])  # 将每次更新过后的w,b记录在history数组中
hisDot = []
hisL = []
rate = 0.005
def L():
    L = 0
    for i in range(len(training_set)):
        if cal(training_set[i]) <= 0:
            w_ = copy.copy(w)
            #w_.append(1)
            w_ = np.sqrt(np.sum(np.array(w_)**2))
            s = -cal(training_set[i])/w_
            L += s
    return round(L, 4)

def update(item):
    """
    随机梯度下降更新参数
    :param item: 参数是分类错误的点
    :return: nothing 无返回值
    """
    global w, b, history  # 把w, b, history声明为全局变量
    hisL.append(L())
    w_ = copy.copy(w)
    w_.append(b)
    print(L(),item, w_)
    w[0] += rate * item[2] * item[0]  # 根据误分类点更新参数,这里学习效率设为1
    w[0] = round(w[0], 4)
    w[1] += rate * item[2] * item[1]
    w[1] = round(w[1], 4)
    b += rate * item[2]
    b = round(b, 4)
    history.append([copy.copy(w), b])  # 将每次更新过后的w,b记录在history数组中
    hisDot.append([item[0], item[1]])
    #print(L())

def cal(item):
    """
    计算item到超平面的距离,输出yi(w*xi+b)
    （我们要根据这个结果来判断一个点是否被分类错了。如果yi(w*xi+b)>0,则分类错了）
    :param item:
    :return:
    """
    res = 0
    for i in range(len(item) - 1):  # 迭代item的每个坐标，对于本文数据则有两个坐标x1和x2
        res += item[i] * w[i]
    r = np.dot([item[0],item[1]], w)
    res += b
    res *= item[2]  # 这里是乘以公式中的yi
    return res


def check():
    """
    检查超平面是否已将样本正确分类
    :return: true如果已正确分类则返回True
    """
    flag = False
    for item in training_set:
        #print("check",item, cal(item))
        if cal(item) < 0:  # 如果有分类错误的
            flag = True  # 将flag设为True
            update(item)  # 用误分类点更新参数
    if not flag:  # 如果没有分类错误的点了
        print("最终结果: w: " + str(w) + "b: " + str(b))  # 输出达到正确结果时参数的值
    return flag  # 如果已正确分类则返回True,否则返回False


if __name__ == "__main__":

    for i in range(100):  # 迭代1000遍
        if not check(): break  # 如果已正确分类，则结束迭代
    # 以下代码是将迭代过程可视化
    # 首先建立我们想要做成动画的图像figure, 坐标轴axis,和plot element
    pltGif_ = pltGif(training_set, history, hisDot)
    pltGif_.save()


    """
    

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    line, = ax.plot([], [], 'g', lw=2)  # 画一条线
    label = ax.text([], [], '')


    def init():
        line.set_data([], [])
        x, y, x_, y_ = [], [], [], []
        for p in training_set:
            if p[2] > 0:
                x.append(p[0])  # 存放yi=1的点的x1坐标
                y.append(p[1])  # 存放yi=1的点的x2坐标
            else:
                x_.append(p[0])  # 存放yi=-1的点的x1坐标
                y_.append(p[1])  # 存放yi=-1的点的x2坐标
        plt.plot(x, y, 'bo', x_, y_, 'rx')  # 在图里yi=1的点用点表示，yi=-1的点用叉表示
        plt.axis([-6, 6, -6, 6])  # 横纵坐标上下限
        plt.grid(True)  # 显示网格
        plt.xlabel('x1')  # 这里我修改了原文表示
        plt.ylabel('x2')  # 为了和原理中表达方式一致，横纵坐标应该是x1,x2
        plt.title('Perceptron Algorithm (www.hankcs.com)')  # 给图一个标题：感知机算法
        return line, label

    def animate(i):
        global history, ax, line, label
        w = history[i][0]
        b = history[i][1]
        if w[1] == 0: return line, label
        # 因为图中坐标上下限为-6~6，所以我们在横坐标为-7和7的两个点之间画一条线就够了，这里代码中的xi,yi其实是原理中的x1,x2
        x1 = -7
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 7
        y2 = -(b + w[0] * x2) / w[1]
        line.set_data([x1, x2], [y1, y2])  # 设置线的两个点
        x1 = 0
        y1 = -(b + w[0] * x1) / w[1]
        label.set_text(history[i])
        label.set_position([x1, y1])
        ss = 'Perceptron Algorithm (www.hankcs.com)%s'%i
        plt.title(ss)  # 给图一个标题：感知机算法
        return line, label


    print("参数w,b更新过程：", history)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=1000, repeat=True, blit=True)
    anim.save('basic_animation.gif', fps=80, extra_args=['-vcodec', 'libx264'])
    #plt.show()
    
    """
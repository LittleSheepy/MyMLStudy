import pylab
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from random import shuffle


def bubblesort_anim(a):
    x = range(len(a))
    swapped = True  # 标记排序是否已经结束
    while swapped:
        plt.clf()  # 清空显示窗口以便实现动画效果
        swapped = False
        for i in range(len(a) - 1):
            if a[i] > a[i + 1]:  # 比较相邻两个元素
                a[i + 1], a[i] = a[i], a[i + 1]  # 如果左边元素比右边元素值大则进行互换
                swapped = True
        plt.plot(x, a, 'k.', markersize=6)  # 将一次冒泡排序后的元素排列变化显示到窗口
        plt.pause(0.01)


def case1():
    a = list(range(4))  # 初始化0-299，总共300个元素
    shuffle(a)  # 将300个元素随机排列
    # 执行冒泡排序将300个元素进行升序排列
    bubblesort_anim(a)









if __name__ == '__main__':
    case1()




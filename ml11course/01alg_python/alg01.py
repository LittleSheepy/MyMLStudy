import pylab
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from random import shuffle

"""
点型：圆圈'O',加号'+',星号'*',点'.',叉号'X',方形's',菱形'd'
线型：实线'-',虚线'--',点线':',点划线'-.'
颜色：'r','g','b','c','m','y','k','w','none'-红、绿、蓝、青、洋红、黄、黑、白、默认
"""
# 冒泡排序
def bubblesort_anim(a):
    x = range(len(a))
    swapped = True  # 标记排序是否已经结束
    while swapped:
        plt.clf()  # 清空显示窗口以便实现动画效果
        swapped = False
        plt.plot(x, a, 'k.', markersize=6)  # 将一次冒泡排序后的元素排列变化显示到窗口
        plt.pause(0.01)
        for i in range(len(a) - 1):
            plt.plot(i, a[i], 'g.', markersize=10)
            if a[i] > a[i + 1]:  # 比较相邻两个元素
                plt.plot([i, i+1], [a[i], a[i + 1]], 'x', markersize=6)  # 将一次冒泡排序后的元素排列变化显示到窗口
                plt.pause(1)
                a[i + 1], a[i] = a[i], a[i + 1]  # 如果左边元素比右边元素值大则进行互换
                swapped = True
                plt.plot([i, i+1], [a[i], a[i + 1]], 'o', markersize=6)  # 将一次冒泡排序后的元素排列变化显示到窗口
                plt.pause(1)
                plt.clf()  # 清空显示窗口以便实现动画效果

                plt.plot(x, a, 'k.', markersize=6)  # 将一次冒泡排序后的元素排列变化显示到窗口
                plt.pause(0.1)
        plt.plot(x, a, 'k.', markersize=6)  # 将一次冒泡排序后的元素排列变化显示到窗口
        plt.pause(0.01)


def case1():
    # a = list(range(5))  # 初始化0-299，总共300个元素
    # shuffle(a)  # 将300个元素随机排列
    a = [6,5,1,2,3,4]
    # 执行冒泡排序将300个元素进行升序排列
    bubblesort_anim(a)









if __name__ == '__main__':
    case1()




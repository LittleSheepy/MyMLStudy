import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class pltGif:
    def __init__(self, dataSet, stepHis, stepDot):
        self.dataSet = dataSet
        self.stepHis = stepHis
        self.stepDot = stepDot
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.line, = self.ax.plot([], [], 'g', lw=2)  # 画一条线
        #self.actDotLine, = self.ax.scatter([0], [0], s=80,c='r',marker='x')   # 画一点
        self.label = self.ax.text([], [], '')
        self.actDotLine = 0
        return
    def init(self):
        self.line.set_data([], [])
        x, y, x_, y_ = [], [], [], []
        for p in self.dataSet:
            if p[2] > 0:
                x.append(p[0])  # 存放yi=1的点的x1坐标
                y.append(p[1])  # 存放yi=1的点的x2坐标
            else:
                x_.append(p[0])  # 存放yi=-1的点的x1坐标
                y_.append(p[1])  # 存放yi=-1的点的x2坐标
        plt.plot(x, y, 'bo', x_, y_, 'rx')  # 在图里yi=1的点用点表示，yi=-1的点用叉表示
        plt.axis([-20, 20, -20, 20])  # 横纵坐标上下限
        plt.grid(True)  # 显示网格
        plt.xlabel('x1')  # 这里我修改了原文表示
        plt.ylabel('x2')  # 为了和原理中表达方式一致，横纵坐标应该是x1,x2
        plt.title('Perceptron Algorithm (www.hankcs.com)')  # 给图一个标题：感知机算法
        return self.line, self.label

    def animate(self,i):
        plt.title('Perceptron Algorithm %s' % i)  # 给图一个标题：感知机算法
        print(i)
        w = self.stepHis[i][0]
        b = self.stepHis[i][1]
        if w[1] == 0:
            return self.line, self.label
        # 因为图中坐标上下限为-6~6，所以我们在横坐标为-7和7的两个点之间画一条线就够了，这里代码中的xi,yi其实是原理中的x1,x2
        x1 = -20
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 20
        y2 = -(b + w[0] * x2) / w[1]
        self.line.set_data([x1, x2], [y1, y2])  # 设置线的两个点
        if self.actDotLine != 0:
            if i+1 <= len(self.stepDot):
                self.actDotLine.remove()
        #self.actDotLine.set_data([self.stepDot[i][0]], [self.stepDot[i][1]])
        if i+1 <= len(self.stepDot):
            self.actDotLine = self.ax.scatter([self.stepDot[i][0]], [self.stepDot[i][1]], s=80, c='r', marker='x')  # 画一点
        else:
            self.actDotLine = self.ax.scatter([0], [0], s=80,c='r',marker='x')   # 画一点
        #self.actDotLine = self.ax.scatter([1], [2], s=80,c='r',marker='x')    # 画一点
        x1 = 0
        y1 = -(b + w[0] * x1) / w[1]
        self.label.set_text(self.stepHis[i])
        self.label.set_position([x1, y1])
        return self.line, self.label

    def save(self):
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init, frames=len(self.stepHis), interval=1000, repeat=True,
                                       blit=True)
        anim.save('basic_animation.gif', fps=80, extra_args=['-vcodec', 'libx264'])
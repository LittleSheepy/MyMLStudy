import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq  # 方法二中使用
from sklearn import linear_model
from sklearn import datasets


x, y = datasets.make_regression(n_samples=100, n_features=1, bias=2.0, noise=0.01, shuffle = False, random_state=1)
x = x.reshape([-1])

# 线性回归模型类
class Linear_regression_methods:
    def __init__(self, x, y):
        self.x = x
        self.X = np.vstack([x, np.ones(len(x))]).T
        self.y = y
        self.data = np.vstack([x,y]).T

    def plt_method(self, title, w, b):
        plt.title(title)
        plt.plot(self.x, self.y, 'o', label='data', markersize=10)
        plt.plot(self.x, w * self.x + b, 'r', label='line')
        plt.legend()
        plt.show()

    def print_method(self, title, a, b):
        return print('-'*50 + "\n{}\ny = {:.5f}x + {:.5f}".format(title, a, b))

    def computer_error(self, a, b):
        x = self.x
        y = self.y
        totalError = (y - (a * x + b)) ** 2
        totalError = np.sum(totalError, axis=0)
        results = totalError / float(len(x))
        return print('this model final error: {:.5f}'.format(results))

    # 调用numpy.linalg.lstsq()方法
    def one_leastsq_call_numpy_pakeage(self):
        A = np.vstack([self.x, np.ones(len(self.x))]).T
        a, b = np.linalg.lstsq(A, self.y, rcond=-1)[0]  # 求一个线性方程组的最小二乘解
        self.print_method('first leastsq_call_numpy_pakeage', a, b)
        self.plt_method('first leastsq_call_numpy_pakeage', a, b)  # 调用画图方法
        self.computer_error(a, b)

    # 调用scipy.optimize中的lestsq方法
    def two_leatsq_call_scipy_pakeage(self):
        def fun(p, x):  # 定义想要拟合的函数
            k, b = p  # 从参数p获得拟合参数
            return k*x + b

        def err(p, x, y):  # 定义误差函数
            return fun(p, x) - y

        # 定义起始的参数 即从 y = 1*x+1 开始，其实这个值可以随便设，只不过会影响到找到最优解的时间
        p0 = [1, 1]  # 也可随机初始化
        # leastsq函数需传入numpy类型
        xishu = leastsq(err, p0, args=(self.x, self.y))
        self.print_method('second leatsq_call_scipy_pakeage', xishu[0][0], xishu[0][1])
        self.plt_method('second leatsq_call_scipy_pakeage', xishu[0][0], xishu[0][1])
        self.computer_error(xishu[0][0], xishu[0][1])

    # 最小二乘法手动实现方法
    def three_leastsq_function(self):
        x = self.x
        y = self.y
        n = len(x)
        sumX, sumY, sumXY, sumXX = 0, 0, 0, 0
        for i in range(0, n):
            sumX += x[i]
            sumY += y[i]
            sumXX += x[i] * x[i]
            sumXY += x[i] * y[i]
        a = (sumXY - (1 / n) * (sumX * sumY)) / (sumXX - (1 / n) * sumX * sumX)
        b = sumY / n - a * sumX / n
        self.print_method('third leastsq_function', a, b)
        self.plt_method('third leastsq_function', a, b)
        self.computer_error(a, b)

    # 最小二乘法手动实现方法 用矩阵
    def three_leastsq_function_matrix(self):
        x = self.X
        y = self.y
        W = np.dot(np.dot(np.matrix(np.dot(x.T, x)).I, x.T),y)  # (X.T X)**-1 X.T Y
        w, b = W[0,0], W[0,1]
        self.print_method('third leastsq_function matrix', w, b)
        self.plt_method('third leastsq_function matrix', w, b)
        self.computer_error(w, b)

    # 用sklearn
    def four_linear_model_call_sklearn(self):
        # train model on data
        body_reg = linear_model.LinearRegression()
        x_values = self.x.reshape(-1, 1)
        y_values = self.y.reshape(-1, 1)
        body_reg.fit(x_values, y_values)
        a,b = body_reg.coef_[0][0], body_reg.intercept_[0]
        self.print_method('fourth linear_model_call_sklearn', a, b)
        self.plt_method('fourth linear_model_call_sklearn', a, b)
        self.computer_error(a, b)

    # 梯度下降法 y = wx + b
    def five_linear_regression(self, lr=0.01, epoches=1000):
        N = float(len(self.X))
        [w, b] = np.ones(2)
        # gradient descent
        for _ in range(epoches):
            y_pre = w * self.x + b
            delt_y = self.y - y_pre
            db = - 2 * delt_y
            db = np.sum(db)/N
            dw = -2 * np.dot(self.x, delt_y)
            dw /= N
            [w,b] = [w,b] - np.dot(lr, [dw, db])
        self.print_method('five_linear_regression', w, b)
        self.plt_method('five_linear_regression', w, b)
        self.computer_error(w, b)

    # 梯度下降法 y = WX
    def six_linear_regression(self, lr=0.01, epoches=1000):
        W = np.ones(len(self.X[0]))
        N = float(len(self.X))
        # gradient descent
        for _ in range(epoches):
            dW = -(2/N)* np.dot(y - np.dot(self.X, W), self.X)
            W -= lr * dW    # 用导数更新W值
        w, b = W
        self.print_method('six_linear_regression', w, b)
        self.plt_method('six_linear_regression', w, b)
        self.computer_error(w, b)

    # 梯度下降法 单个样本
    def seven_linear_regression(self, lr=0.01, num_iter=1000):
        W = np.ones(len(self.X[0]))
        # gradient descent
        for i in range(num_iter):
            # this gradient step
            for xi, yi in zip(self.X, self.y):
                W_gradient = -2 * xi * (yi - np.dot(xi,W))  # 目的是极小化平方误差
                W -= lr * W_gradient
        w, b = W
        self.print_method('seven_linear_regression', w, b)
        self.plt_method('seven_linear_regression', w, b)
        self.computer_error(w, b)


model = Linear_regression_methods(x, y)
model.one_leastsq_call_numpy_pakeage()
model.two_leatsq_call_scipy_pakeage()
model.three_leastsq_function()
model.three_leastsq_function_matrix()
model.four_linear_model_call_sklearn()
model.five_linear_regression()
model.six_linear_regression()
model.seven_linear_regression()
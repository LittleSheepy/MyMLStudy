import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0] * x[0] + 50 * x[1] * x[1]

def g(x):
    return np.array([2 * x[0], 100 * x[1]])

xi = np.linspace(-200, 200, 1000)
yi = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(xi, yi)
Z = X * X + 50 * Y * Y

#% matplotlib inline
def contour(X, Y, Z, arr=None):
    plt.figure(figsize=(15, 7))
    xx = X.flatten()
    yy = Y.flatten()
    zz = Z.flatten()
    plt.contour(X, Y, Z, colors='black')
    plt.plot(0, 0, marker='*')
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i + 2, 0], arr[i:i + 2, 1])
    plt.show()


#contour(X, Y, Z)
def gd(x_start, step, g):  # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    for i in range(50):
        grad = g(x)
        x -= grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {00} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot

def momentum(x_start, step, g, discount=0.7):  # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        grad = g(x)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break;
    return x, passing_dot

def nesterov(x_start, step, g, discount=0.7):  # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        x_future = x - step * discount * pre_grad
        grad = g(x_future)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 0.1:
            break;
    return x, passing_dot
minGrad = 100
def RMSProp(x_start, step, g, discount=0.7):  # gd代表了Gradient Descent
    beta = 0.9
    epsilon = 1e-8
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    Sdw = 0
    global minGrad
    for i in range(2000):
        grad = g(x)
        Sdw = beta * Sdw + (1 - beta) * grad* grad
        Sgrad = grad/(np.sqrt(Sdw) + epsilon)
        x -= Sgrad * step

        passing_dot.append(x.copy())

        if minGrad > abs(sum(grad)):
            minGrad = abs(sum(grad))
        print('[ Epoch {0} ] grad = {1},Sdw = {2},Sgrad = {3}, x = {3}'.format(i, grad, Sdw, Sgrad, x))

        if abs(sum(grad)) < 0.1:
            break;
    return x, passing_dot


def Adam(x_start, step, g, discount=0.7):  # gd代表了Gradient Descent
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    Sdw = 0
    m = 0
    v = 0
    global minGrad
    for i in range(2000):
        grad = g(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_ = m / (1 - beta1)
        v_ = v / (1 - beta2)

        Sgrad = m_/(np.sqrt(v_) + epsilon)

        x -= Sgrad * step

        passing_dot.append(x.copy())

        if minGrad > abs(sum(grad)):
            minGrad = abs(sum(grad))
        print('[ Epoch {0} ] grad = {1},Sdw = {2},Sgrad = {3}, x = {3}'.format(i, grad, Sdw, Sgrad, x))

        if abs(sum(grad)) < 0.1:
            break;
    return x, passing_dot


res, x_arr = Adam([150, 75], 0.1, g)
contour(X, Y, Z, x_arr)

print(minGrad)
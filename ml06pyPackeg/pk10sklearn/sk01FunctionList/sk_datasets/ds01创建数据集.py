import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

"""
factor: 内圈和外圈半径之比
"""
# 创建两个同心圆
def my_make_circles():
    X, y = make_circles(n_samples=100, noise=0.05, factor=0.3, shuffle=False)
    X1 = X[y == 0]
    X2 = X[y == 1]

    fig, ax = plt.subplots()
    ax.scatter(X1[:, 0], X1[:, 1], color='blue', label='set 1')
    ax.scatter(X2[:, 0], X2[:, 1], color='orange', label='set 2')
    ax.legend(loc=0)
    plt.show()
    return X1, X2


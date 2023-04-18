# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics.pairwise import pairwise_distances

from Sinkhorn_Knopp import compute_optimal_transport, compute_optimal_transport_test

def creat_data():
    # two concentric circles
    X, y = make_circles(n_samples=100, noise=0.05, factor=0.5, shuffle=False)
    X1 = X[y==0]
    X2 = X[y==1]
    return X1, X2

def show_data(*args, show = True):
    color = ['blue', 'orange', 'red', 'gray', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']
    fig, ax = plt.subplots()
    for i, X in enumerate(args):
        ax.scatter(X[:, 0], X[:, 1], color=color[i], label=f'set {i+1}')
    ax.legend(loc=0)
    if show:
        plt.show()
    else:
        return ax

def show_matching(X1, X2, ax, P):
    m, n = len(X2), len(X1)
    for i in range(n):
        for j in range(m):
            if P[i, j] > 1e-5:
                ax.plot([X1[i, 0], X2[j, 0]], [X1[i, 1], X2[j, 1]], alpha=P[i, j] * n, color='red')
    plt.show()

def run_optimal_transport(X1, X2, lam=100, show=True):
    M = pairwise_distances(X1, X2, metric='euclidean')
    # Uniform weights
    n, m = M.shape
    r = np.ones(n) / n
    c = np.ones(m) / m
    P, d = compute_optimal_transport(M, r, c, lam=lam, epsilon=1e-6)
    if show:
        plt.imshow(M, cmap="gray")
        plt.show()
        plt.imshow(P, cmap="gray")
        plt.show()
    return P, d, r, c

"""
Interpolate between the two distributions.
在两个分布之间进行插值。
Input:
    - alpha : value between 0 and 1 for the interpolation
Output:
    - X : the interpolation between X1 and X2
    - w : weights of the points
"""
def InterpolateBetweenDistributions(alpha = 0.6):
    mixing = P.copy()
    # Normalize, so each row sums to 1 (i.e. probability) 归一化，因此每一行之和为1(即概率)
    mixing /= r.reshape((-1, 1))
    X = (1 - alpha) * X1 + alpha * mixing @ X2
    w = (1 - alpha) * r + alpha * mixing @ c
    return X

if __name__ == '__main__':
    X1, X2 = creat_data()
    show_data(X1, X2)
    P, d, r, c = run_optimal_transport(X1, X2)
    ax = show_data(X1, X2, show=False)
    show_matching(X1, X2, ax, P)
    X = InterpolateBetweenDistributions()
    show_data(X1, X2, X)




# 方法2：
# Sinkhorn-Knopp算法
'''
    https://michielstock.github.io/posts/2017/2017-11-5-OptimalTransport/
    https://zhuanlan.zhihu.com/p/542379144
'''
# numpy
def compute_optimal_transport(scores, eps = 5, n_iter = 3):
    """
    Computes the optimal transport matrix and Sinkhorn distance using the
    Sinkhorn-Knopp algorithm
    Inputs:
        - scores : cost matrix (n * m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - eps : strength of the entropic regularization
        - epsilon : convergence parameter
    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    t0 = time.time()
    n, m = scores.shape
    r = np.ones(n)  # P矩阵列和为r
    c = np.ones(m)*(n/m)  # P矩阵行和为c
    # 确保：np.sum(r)==np.sum(c)
    P = np.exp(- scores / eps)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    # while np.max(np.abs(u - P.sum(1))) > epsilon:
    for _ in range(n_iter):
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1)) # 行归r化
        P *= (c / P.sum(0)).reshape((1, -1)) # 列归c化
    t1 = time.time()
    compute_time = t1 - t0
    dist = np.sum(P * scores)
    return P, dist, compute_time

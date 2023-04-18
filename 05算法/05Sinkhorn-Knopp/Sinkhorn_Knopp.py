import numpy as np

def compute_optimal_transport(M, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the Sinkhorn-Knopp algorithm
    利用Sinkhorn-Knopp算法计算最优传输矩阵和Slinkhorn距离
    Inputs:
        - M : cost matrix (n x m)                       成本矩阵
        - r : vector of marginals (n, )                 成本向量
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization 熵正则化的强度
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)          最优传输矩阵
        - dist : Sinkhorn distance                      Slinkhorn距离
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    # Avoiding poor math condition
    P /= P.sum()
    u = np.zeros(n)
    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        # Shape (n, )
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * M)

def compute_optimal_transport_test(M, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the Sinkhorn-Knopp algorithm
    利用Sinkhorn-Knopp算法计算最优传输矩阵和Slinkhorn距离
    Inputs:
        - M : cost matrix (n x m)                       成本矩阵
        - r : vector of marginals (n, )                 成本向量
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization 熵正则化的强度
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)          最优传输矩阵
        - dist : Sinkhorn distance                      Slinkhorn距离
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    print(P)
    # Avoiding poor math condition 避免糟糕的数学状况
    P /= P.sum()
    u = np.zeros(n)
    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    print(P)
    print("c P\n", np.round(np.array(P), 3))
    while True:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        P_sum1 = P.sum(1)
        u_P_sum1 = u - P_sum1
        abs_u_P_sum1 = np.abs(u_P_sum1)
        max_abs_u_P_sum1 = np.max(abs_u_P_sum1)
        #print("max_abs_u_P_sum1 = ", max_abs_u_P_sum1)
        if max_abs_u_P_sum1 < epsilon:
            break
        # Shape (n, )
        u = P.sum(1)
        print("r sum:\n", str(np.round(u, 3)))
        r_u = r / u
        r_u_reshape = r_u.reshape((-1, 1))
        print("r r_u_reshape:\n", str(np.round(r_u_reshape, 3)))
        P_old1 = P.copy()
        P = P * r_u_reshape
        print("r P\n", np.round(np.array(P), 3))
        # P *= (r / u).reshape((-1, 1))

        P_sum0 = P.sum(0)
        print("c sum:\n", str(np.round(P_sum0, 3)))
        c_P_sum0 = c / P_sum0
        c_P_sum0_reshape = c_P_sum0.reshape((1, -1))
        print("r c_P_sum0_reshape:\n", str(np.round(c_P_sum0_reshape, 3)))
        P_old2 = P.copy()
        P = P * c_P_sum0_reshape
        print("c P\n", np.round(np.array(P), 3))
        # P *= (c / P.sum(0)).reshape((1, -1))
        pass
    return P, np.sum(P * M)
if __name__ == '__main__':
    pass
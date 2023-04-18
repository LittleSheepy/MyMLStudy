# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split

from Sinkhorn_Knopp import compute_optimal_transport
from SinkhornKnopp应用实例_分布匹配 import show_data, run_optimal_transport, show_matching

def creat_data():
    Xtr, Xte, ytr, yte = train_test_split(*make_blobs(n_samples=200, centers=3, cluster_std=1.5), test_size=0.5)
    Xte += np.random.randn(*Xte.shape) * 2 + np.array([[2, -3]])
    return Xtr, Xte, ytr, yte

if __name__ == '__main__':
    Xtr, Xte, ytr, yte = creat_data()
    show_data(Xtr[ytr == 0, :], Xtr[ytr == 1, :], Xtr[ytr == 2, :], Xte)

    P, d, r, c = run_optimal_transport(Xtr, Xte, 50)

    ax = show_data(Xtr[ytr == 0, :], Xtr[ytr == 1, :], Xtr[ytr == 2, :], Xte, show=False)
    show_matching(Xtr, Xte, ax, P)

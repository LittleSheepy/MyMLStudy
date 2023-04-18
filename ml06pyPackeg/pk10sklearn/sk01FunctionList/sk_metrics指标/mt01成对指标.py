from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

from sk_datasets.ds01创建数据集 import my_make_circles

# 计算距离
def my_pairwise_distances():
    X1, X2 = my_make_circles()      # 创建两个同心圆
    M = pairwise_distances(X1, X2, metric='euclidean')
    plt.imshow(M, cmap="gray")
    plt.show()
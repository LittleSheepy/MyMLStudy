import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


print("---make_regression---"*5)
x, y = datasets.make_regression(n_samples=100, n_features=1,  noise=5, shuffle = False)
print(f"x.shape={x.shape}, y.shape={y.shape}")
plt.scatter(x, y)
plt.show()

print("---make_blobs---"*5)
"""
def make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None):
centers: 生成样本的中心数， 即类别数目
cluster_std: 每个类别方差
shuffle: 是否将样本打乱
random_state: 随机种子
"""
x, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=[1.0,3.0],
                           center_box=(-10.0, 10.0), shuffle=False)
print(f"x.shape={x.shape}, y.shape={y.shape}")
plt.scatter(x[:,0], x[:,1], s=6, c=y)
plt.show()

print("---make_circles---"*5)
"""
def make_circles(n_samples=100, 
                 shuffle=True,  # 打乱
                 noise=None, 
                 random_state=None,
                 factor=.8):
"""
x, y = datasets.make_circles(n_samples=1000, shuffle=True, noise=0.03, random_state=None, factor=.8)
print(f"x.shape={x.shape}, y.shape={y.shape}")
plt.scatter(x[:,0], x[:,1], s=6, c=y)
plt.show()

print("---make_classification---"*5)
"""
def make_classification(n_samples=100, n_features=20, 
                        n_informative=2, # 有价值特征
                        n_redundant=2,   # 冗余特征个数（有效特征的随机组合）
                        n_repeated=0,    # 重复特征个数（有效和冗余随机组合）
                        n_classes=2,     # 类别
                        n_clusters_per_class=2,  # 簇个数
                        weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):
                        
"""
from mpl_toolkits.mplot3d import Axes3D
x, y = datasets.make_classification(n_samples=500, n_features=3, n_informative=3,
                          n_redundant=0, n_repeated=0, n_classes=3,
                          n_clusters_per_class=1, weights=None,
                          flip_y=0.01, class_sep=1.0, hypercube=True,
                          shift=0.0, scale=1.0, shuffle=True, random_state=None)
print(f"x.shape={x.shape}, y.shape={y.shape}")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], s=6,c=y)
plt.show()

print("---make_moons---"*5)
"""
def make_moons(n_samples=100, shuffle=True, noise=None, random_state=None):
"""
x,y = datasets.make_moons(n_samples=100, shuffle=True, noise=0.06, random_state=None)
print(f"x.shape={x.shape}, y.shape={y.shape}")
plt.scatter(x[:,0], x[:,1], c=y, s=7)
plt.show()








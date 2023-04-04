import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

"""
波士顿房价数据
样本shape=(506,13)
x=[6.320e-03 1.800e+01 2.310e+00 0.000e+00 5.380e-01 6.575e+00 6.520e+01 4.090e+00 1.000e+00 2.960e+02 1.530e+01 3.969e+02 4.980e+00]
y=24.0
"""
print("---load_boston---"*5)
boston = datasets.load_boston()
data = boston.data
target = boston.target
feature_names = boston.feature_names
print(feature_names)
print(f"data.shape={data.shape}\ndata[0]=\n{data[0]}")
print(f"target.shape={target.shape},target[0]={target[0]}")

"""
iris花卉数据
样本shape=(150,4), 类别个数lable=3
"""
print("---load_iris---"*5)
iris = datasets.load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names
print(feature_names)
print(f"data.shape={data.shape}\ndata[0]=\n{data[0]}")
print(f"target.shape={target.shape},target[0]={target[0]}")

"""
糖尿病数据集 回归使用
样本shape=(442,10)
"""
print("---diabetes---"*5)
diabetes = datasets.load_diabetes()
data = diabetes.data
target = diabetes.target
feature_names = diabetes.feature_names
print(feature_names)
print(f"data.shape={data.shape}\ndata[0]=\n{data[0]}")
print(f"target.shape={target.shape},target[0]={target[0]}")

"""
手写体数据集 分类使用
"""
print("---digits---"*5)
digits = datasets.load_digits()
print(dir(digits))
data = digits.data          # (1797, 64)
images = digits.images      # (1797, 8, 8)
target = digits.target
target_names = digits.target_names
print(f"data.shape={data.shape}\ndata[0]=\n{data[0]}")
print(f"target.shape={target.shape},target[0]={target[0]}")
plt.imshow(images[1], cmap="gray")
plt.title(f"{target[1]}")
plt.show()

"""
linnerud体能训练数据集，多元回归使用。
"""
print("---linnerud---"*5)
linnerud = datasets.load_linnerud()
print(dir(linnerud))
data = linnerud.data          # (20, 3)
target = linnerud.target      # (20, 3)
target_names = linnerud.target_names
print(f"data[0]={data[0]}")
print(f"target[0]={target[0]}")

china = datasets.load_sample_image("china.jpg")
tiger = datasets.load_sample_image("tiger.jpg")
print(china.shape)
plt.imshow(china)
plt.show()
print(tiger.shape)
plt.imshow(tiger)
plt.show()

















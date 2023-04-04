import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import cv2

l = np.array([i for i in range(10)])
l1 = np.reshape(l, [-1,5])
l2 = np.reshape(l, [5,-1])
l2 = np.transpose(l2, [1,0])


path = 'D:/ML_datas/MNIST_data'
ds = read_data_sets(path)
print(ds.train.num_examples)
print(ds.test.num_examples)
print(ds.validation.num_examples)

def reshape1(imgs):
    imgs = np.reshape(imgs, [-1, 28, 28])
    imgs = np.transpose(imgs, [1, 0, 2])  # [28, -1, 28]
    imgs = np.reshape(imgs, [28, -1, 28 * 10])
    imgs = np.transpose(imgs, [1, 0, 2])  # [-1, 28, 28*10]
    imgs = np.reshape(imgs, [-1, 28 * 10])
    return imgs

"""
01234
56789
"""
def reshape2(imgs):
    imgs = np.reshape(imgs, [-1, 10, 28, 28])
    imgs = np.transpose(imgs, [0, 2, 1, 3])  # [28, -1, 28]
    imgs = np.reshape(imgs, [-1, 28 * 10])
    return imgs

"""
02468
13579
"""
def reshape3(imgs):
    imgs = np.reshape(imgs, [10, -1, 28, 28])
    imgs = np.transpose(imgs, [1, 2, 0, 3])  # [28, -1, 28]
    imgs = np.reshape(imgs, [-1, 28 * 10])
    return imgs

imgs, labels = ds.train.next_batch(20)    # [-1, 784]

imgs1 = reshape1(imgs)
print(imgs1)
print(labels)
plt.imshow(imgs1)
plt.show()

imgs2 = reshape2(imgs)
plt.imshow(imgs2)
plt.show()

imgs3 = reshape3(imgs)
plt.imshow(imgs3)
plt.show()
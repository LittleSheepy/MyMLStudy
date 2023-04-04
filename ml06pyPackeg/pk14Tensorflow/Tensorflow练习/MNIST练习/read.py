import tensorflow as tf
import numpy as np

print(tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
ds = read_data_sets("MNIST_data")
print(ds.train.num_examples)
print(ds.validation.num_examples)
print(ds.test.num_examples)
print("训练集个数：", ds.train.images.shape, ds.train.labels.shape)
print("验证集个数：", ds.validation.images.shape, ds.validation.labels.shape)
print("测试集个数：", ds.test.images.shape, ds.test.labels.shape)



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
ds = read_data_sets("MNIST_data")
train = ds.train
xs, ys = train.next_batch(2)
print(ys)

ax1 = plt.subplot(121)
ax1.set_title(f"{ys[0]}")
plt.imshow(xs[0].reshape(28,28), "gray")
ax1 = plt.subplot(122)
ax1.set_title(f"{ys[1]}")
plt.imshow(xs[1].reshape(28,28), "gray")
plt.show()

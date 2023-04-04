import tensorflow as tf
import numpy as np

print(tf.__version__)

tf.layers.max_pooling2d
print("---tf.layers.max_pooling2d---"*5)
"""
def max_pooling2d(inputs,
                  pool_size, # 3  (3,3)  [3,3]
                  strides,
                  padding='valid',  #valid:    same:
                  data_format='channels_last',
                  name=None):

"""
x = tf.random_normal([2, 4, 4, 1])
pool1 = tf.layers.max_pooling2d(x, 2, 1, 'same')
pool2 = tf.layers.max_pooling2d(x, 2, 1, 'valid')
with tf.Session() as session:
    x_v = session.run(x)
    print("x_shape ",x.shape)
    pool1_v = session.run(pool1)
    print("p1_shape",pool1_v.shape)
    pool2_v = session.run(pool2)
    print("p1_shape",pool2_v.shape)


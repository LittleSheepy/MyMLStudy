import tensorflow as tf
import numpy as np

print(tf.__version__)

x = tf.placeholder(tf.float32, [3, 7, 5, 2, 8])

y = tf.transpose(x, [0, 1, 3, 4, 2])  # [3, 7, 2, 8, 5]
z = tf.reshape(y, [3, 112, 5])
w = tf.expand_dims(x,axis=0)
print(y.shape)
print(z.shape)
print(w.shape)

import tensorflow.compat.v1 as tf
import numpy as np
print(tf.__version__)


print("---tf1.layers.dense---"*5)
a = tf.random.normal([2,2,2,3])
d1 = tf.layers.dense(a, 9)
print("d1",d1.shape)

w = tf.random.normal([3,9])
b = tf.random.normal([9])
d2 = tf.matmul(a, w)+b
print("d2",d2.shape)

"""
class Dense(Layer):
    def __init__():
    def call(self, inputs):
    

  def __init__(self,
               units,  输出空间的维数
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
"""
print("---keras.layers.Dense---"*5)
a = tf.random.normal([2,3])
D1 = tf.keras.layers.Dense(9)
Out_D1 = D1(a)
print("Out_D1",Out_D1.shape)


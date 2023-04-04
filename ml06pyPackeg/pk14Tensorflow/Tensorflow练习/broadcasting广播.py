import tensorflow as tf
import numpy as np

print(tf.__version__)

# broadcasting
# 0) 标量可以和任意矩阵运算
# 1) a.shape == b.shape
# 2) a.shape == b.shape[:i] + [1] + b.shape[:i]   [2,3,4]  [2,1,4]
c3 = tf.random_uniform([2, 3, 4], 0,1)
c4 = tf.random_uniform([2, 1, 4], 10, 100)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    a3, a4, a34 = session.run([c3, c4, c3 + c4])
    print("a3=\n",a3)
    print("a4=\n",a4)
    print("a34=\n",a34)

# 3) a.shape == b.shape[1:]
c3 = tf.random_uniform([2, 3, 4], 0,1)
c4 = tf.random_uniform([3, 4], 10, 100)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    a3, a4, a34 = session.run([c3, c4, c3 + c4])
    print("a3=\n",a3)
    print("a4=\n",a4)
    print("a34=\n",a34)



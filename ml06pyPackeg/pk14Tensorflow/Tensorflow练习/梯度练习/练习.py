import numpy as np
import tensorflow as tf

print(tf.__version__)


"""
y = abs(x) + xx.T
"""
# x = tf.Variable(initial_value=[[1,-2]], dtype=tf.float32, name="x")
x = tf.constant([[1,-2]], dtype=tf.float32)
one = tf.constant([[1,1]], dtype=tf.float32)
dy = tf.Variable(initial_value=[[-2],[7]], dtype=tf.float32,name="dy")
abs_x = tf.abs(x,"abs_x")               # [1, 2]        [[1, 2]]
x_t=tf.transpose(x,name="x_t")             # (2, 1)        [[1],[-2]]
x_x_t = tf.matmul(x,x_t,name="x_x_t")        # (1, 1)        [[5]]
x_x_t_t = tf.matmul(x_x_t,one,name="x_x_t_t")  # (1, 2)        [[5],[5]]
y = abs_x + x_x_t_t             # (1, 2)        [[6,7]]
loss = tf.matmul(y,dy)
#loss = dy * y


opt = tf.train.GradientDescentOptimizer(0.1)
grads_vals = opt.compute_gradients(loss,[x_x_t])     # [(grad, var), ]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(grads_vals))

import numpy as np
import tensorflow as tf
import pprint

print(tf.__version__)


x = tf.Variable(initial_value=[[1,2],[3,4]], dtype='float32', name="y")
x_e = tf.expand_dims(x, axis=0)
pad = tf.pad(x_e, paddings=[[0,0],[1,1],[1,1]])
w = tf.Variable(initial_value=2., dtype='float32', name="w")
y = w*pad

opt = tf.train.GradientDescentOptimizer(0.1)
grads_vals = opt.compute_gradients(y, [x, pad])     # [(grad, var), ]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    pprint.pprint(sess.run(grads_vals))

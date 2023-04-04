import numpy as np
import tensorflow as tf

print(tf.__version__)

norm_dim = tf.range(1, 2)

with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    norm_dim_v = sess.run(norm_dim)
    print(norm_dim_v)

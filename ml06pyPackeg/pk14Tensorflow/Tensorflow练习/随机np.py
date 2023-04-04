import numpy as np
import tensorflow as tf

print(tf.__version__)

n = np.random.random()
n_tensor = tf.cast(n, tf.float32)
n_tensor2 = tf.random_uniform([])
print(n)

with tf.Session() as sess:
    for i in range(5):
        print(sess.run([n_tensor, n_tensor2]))

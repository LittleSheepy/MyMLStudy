import numpy as np
import tensorflow as tf

print(tf.__version__)

inputs = []
for i in range(2):
    input = tf.placeholder(dtype=tf.float32, shape=[])
    inputs.append(input)


with tf.Session() as sess:
    inputs_data = [[2], [3]]
    inputs_v = sess.run(inputs, {inputs:inputs_data})
    print(inputs_v)


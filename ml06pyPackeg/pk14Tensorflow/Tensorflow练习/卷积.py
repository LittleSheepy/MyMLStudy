import numpy as np
import tensorflow as tf

print(tf.__version__)

"""
[[[[1,2],[1,2]],
  [[1,2],[1,2]]]]
"""
inputs = np.array([[[[1,2],[1,2]],[[1,2],[1,2]]]])
input_data = tf.Variable(inputs, dtype=np.float32)
filter_data = tf.Variable(np.ones([2, 2, 2, 2]), dtype=np.float32)

input_data1 = tf.Variable(inputs, dtype=np.float32)
filter_data1 = tf.Variable(np.ones([2, 2, 2, 2]), dtype=np.float32)

y_c = tf.nn.conv2d(input_data1, filter_data1, strides=[1,1,1,1], padding='SAME')

y_d = tf.nn.depthwise_conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
"""
[[[[4. 4. 8. 8.],[2. 2. 4. 4.]]
  [[2. 2. 4. 4.],[1. 1. 2. 2.]]]]
[1 2 2 4]
"""

depthwise_filter = tf.Variable(np.ones([2, 2, 2, 2]), dtype=np.float32)
pointwise_filter = tf.Variable(np.ones([1, 1, 4, 2]), dtype = np.float32)  # 卷积核两项必须是1
# out_channels >= channel_multiplier * in_channels
y_s = tf.nn.separable_conv2d(input_data, depthwise_filter, pointwise_filter, strides = [1, 1, 1, 1], padding = 'SAME')

init = tf.initialize_all_variables()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    y_c_v = sess.run(y_c)
    y_d_v = sess.run(y_d)
    y_s_v = sess.run(y_s)
    print(y_c_v, y_c_v.shape)
    print(y_d_v, y_d_v.shape)
    print(y_s_v, y_s_v.shape)




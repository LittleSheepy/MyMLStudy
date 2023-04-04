import tensorflow as tf
import numpy as np

print(tf.__version__)

# 创建一个二维变量,默认执行CONSTANT填充
vct = tf.Variable(tf.ones([2, 3]), name="vct")
# 指定填充方式,
pad1 = np.array([[1, 1], [1, 1]])
pad2 = np.array([[1, 3], [3, 3]])
# tf.pad进行填充
vct_pad1 = tf.pad(vct, pad1, name='pad_1')
vct_pad2 = tf.pad(vct, pad2, name='pad_2')
# 创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(vct))
    vp1 = sess.run(vct_pad1)
    print(sess.run(vct_pad2))

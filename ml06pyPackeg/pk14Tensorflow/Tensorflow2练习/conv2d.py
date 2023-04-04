import tensorflow as tf

a = tf.random_uniform([1, 28, 28, 1])
b = tf.layers.conv2d(a, 64, kernel_size=(3,3), strides=(3,3), padding="valid")
c = tf.layers.conv2d(a, 64, kernel_size=(3,3), strides=(3,3), padding="same")
filter = tf.Variable(tf.random_normal([3,3,1,64]))
d = tf.nn.conv2d(a, filter, strides=(1,3,3,1), padding="SAME")
e = tf.nn.conv2d(a, (3,3,1,64), strides=(1,3,3,1), padding="SAME")


print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)








"""
def conv2d(inputs,
           filters, # 整数，表示输出空间的维数（即卷积过滤器的数量）
           kernel_size,
           strides=(1, 1),
           padding='valid',
           data_format='channels_last',
           dilation_rate=(1, 1),
           activation=None,
           use_bias=True,
           kernel_initializer=None,
           bias_initializer=init_ops.zeros_initializer(),
           kernel_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           reuse=None)
"""



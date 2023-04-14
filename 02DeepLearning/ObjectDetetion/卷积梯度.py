import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print()

print(tf.__version__)

x = tf.get_variable("x", [1, 416, 416, 1], dtype=tf.float32,initializer=tf.ones_initializer)    #
convs = []
c = x
c = tf.layers.conv2d(c, 1, 3, 2, padding="SAME", kernel_initializer=tf.ones_initializer)
c = tf.layers.conv2d(c, 1, 3, 2, padding="SAME", kernel_initializer=tf.ones_initializer)
c = tf.layers.conv2d(c, 1, 3, 2, padding="SAME", kernel_initializer=tf.ones_initializer)
c = tf.layers.conv2d(c, 1, 3, 2, padding="SAME", kernel_initializer=tf.ones_initializer)
c = tf.layers.conv2d(c, 1, 3, 2, padding="SAME", kernel_initializer=tf.ones_initializer)
c = tf.layers.conv2d(c, 1, 3, 2, padding="SAME", kernel_initializer=tf.ones_initializer)
# for i in range(3):
#     c = tf.layers.conv2d(c, 1, 3, 2, padding="SAME", kernel_initializer=tf.ones_initializer)
#     convs.append(c)

y = c       # (1, 2, 2, 1)
opt = tf.train.GradientDescentOptimizer(0.1)
grads_vals = opt.compute_gradients(y[0][2][3],[x])     # [(grad, var), ]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
vas = tf.global_variables()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    grads_vals_v = sess.run(grads_vals)
    img = grads_vals_v[0][0][0, :, :, 0]
    img = np.clip(img, 0,1)

    plt.imshow(img, cmap="gray")
    plt.grid()
    plt.show()
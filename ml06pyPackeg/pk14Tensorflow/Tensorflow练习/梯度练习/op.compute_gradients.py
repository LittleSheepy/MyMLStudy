import numpy as np
import tensorflow as tf
"""
wx=20*50=100
loss = 0.5*(w*x-100)**2
dloss_x = w*(wx-100)
"""
print(tf.__version__)

x_input = tf.placeholder(tf.float32, (), "x_input")

#x = tf.Variable(initial_value=10., dtype='float32', name="x")
x = x_input
w = tf.Variable(initial_value=2., dtype='float32', name="y")
y = w*x
loss = 0.5*(y-10)**2

opt = tf.train.GradientDescentOptimizer(0.1)
grads_vals = opt.compute_gradients(loss, [x])     # [(grad, var), ]
# grads = tf.gradients(loss, [x])             # [grad, ]
#
# train_op = opt.apply_gradients(grads)     # [(grad, var), ]
# grads_vals_clip = [[]]
# for i, (g, v) in enumerate(grads_vals):
#     if g is not None:
#         grads_vals[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
#train_op = opt.apply_gradients(grads_vals)
#train_op_mini = opt.minimize(loss, var_list=[x])

#x = x - 0.1 * grads_vals[0][0]
#x_new = x - 0.1 * grads_vals[0][0]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print(sess.run(grads_vals))
    # print(sess.run(grad2))
    x_in = 10.0
    delt = 0
    for i in range(3):
        # print(sess.run(grads_vals))
        # _= sess.run(train_op_mini)
        # _= sess.run(train_op)
        # print(sess.run(x))
        # print(sess.run(x_new))
        # print(sess.run(x.assign(x_new)))
        grads_vals_v = sess.run(grads_vals, {x_input:x_in})
        delt = delt + -0.1*grads_vals_v[0][0]
        x_in = delt + 10
        print(delt, x_in)







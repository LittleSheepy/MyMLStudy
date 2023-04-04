import tensorflow as tf
import numpy as np

print(tf.__version__)


x = tf.get_variable('x', [], dtype=tf.float32)  #, initializer=tf.initializers.ones
x1 = x.assign(x+1)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    x_v, x1_v = session.run([x,x1])
    saver = tf.train.Saver()
    saver.save(session, "model/")
print("x_v",x_v)
print("x1_v",x1_v)

with tf.Session() as session:
    saver = tf.train.Saver()
    saver.restore(session, "model/")
    x_v, x1_v = session.run([x,x1])

print("x_v",x_v)
print("x1_v",x1_v)






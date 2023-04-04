import tensorflow.compat.v1 as tf
print(tf.__version__)

"""
def random_uniform(shape,
                   minval=0,
                   maxval=None,
                   dtype=dtypes.float32,
                   seed=None,
                   name=None):
"""
print("---random_uniform---"*10)
rand = tf.random_uniform([2,3], 0, 1)
with tf.Session() as session:
    print(session.run(rand))


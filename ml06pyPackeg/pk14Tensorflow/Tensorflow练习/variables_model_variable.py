import numpy as np
import tensorflow as tf

print(tf.__version__)

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import variable_scope
#from tensorflow.contrib.layers.python.layers import utils
import tf_slim as slim
from tf_slim.layers import utils


fc = tf.contrib.layers.fully_connected

x = tf.placeholder(tf.float32, [None, 10])
a = fc(x, 2)
b = fc(a, 2)
aa = slim.fully_connected(b, 2)
vars = tf.trainable_variables()
for var in vars:
    print (var.name)

print("*"*20)
def test(input, scope=None, reuse=None):
    with variable_scope.variable_op_scope([input], scope, 'test', reuse=reuse):
        return variables.model_variable('asdf', [1, 1],
                initializer=tf.constant_initializer(0.),
                trainable=True)

c = test(x)
d = test(x)
e = tf.get_variable("e", [2, 3], tf.float32)
with variable_scope.variable_scope("e1"):
    e1 = tf.get_variable("e", [2, 3], tf.float32)
e2 = slim.model_variable("e2", 3)
e3 = slim.model_variable("e3", e1.get_shape()[-1:])
f = tf.model_variables()

# utils.get_variable_collections()
# tf.get_all_collection_keys()

"""
def get_var(scope=None):
    with variable_scope.variable_scope(scope, 'test'):
        return tf.get_variable('temp',
                [10, 10],
                initializer=tf.constant_initializer(0.0),
                trainable=True)
c = get_var()
"""

vars = tf.trainable_variables()
for var in vars:
    print (var.name)

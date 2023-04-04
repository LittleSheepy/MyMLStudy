import tensorflow as tf
import numpy as np

print(tf.__version__)


labels = tf.constant([0,0,0],shape=[1,3], dtype=tf.float32)
logits = tf.constant([3,3,1],shape=[1,3], dtype=tf.float32)
"""
  x = logits, z = labels

  z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
= z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
= z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
= (1 - z) * x + log(1 + exp(-x))
= x - x * z + log(1 + exp(-x))
"""
a3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

logits_sigmoid = tf.nn.sigmoid(logits)
a_gongshi = -(labels * tf.log(logits_sigmoid) + (1-labels)*tf.log((1-logits_sigmoid)))

a4 = tf.nn.weighted_cross_entropy_with_logits




with tf.Session() as sess:
    a3_v = sess.run(a3)
    logits_sigmoid_v,a_gongshi_v = sess.run([logits_sigmoid,a_gongshi])
    print(a3_v)
    print(logits_sigmoid_v, a_gongshi_v)

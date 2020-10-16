"""
Implement some custom layers, not provided by TensorFlow.
"""
import numpy as np
import tensorflow as tf

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

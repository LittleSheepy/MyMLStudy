import numpy as np
import tensorflow as tf

print(tf.__version__)

def generate_grids(feature_shape):
    y = tf.tile(tf.range(feature_shape, dtype=tf.int32)[:, tf.newaxis], [1, feature_shape])
    x = tf.tile(tf.range(feature_shape, dtype=tf.int32)[tf.newaxis, :], [feature_shape, 1])
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)  # (?, ?, 2) (52, 52, 2)
    return tf.cast(xy_grid, tf.float32)

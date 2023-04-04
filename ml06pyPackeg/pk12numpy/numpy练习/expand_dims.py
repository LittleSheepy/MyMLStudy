import numpy as np
import tensorflow as tf

print(tf.__version__)

a = np.arange(6).reshape([2,3])
ae = np.expand_dims(a, axis=-1)
print(a.shape)

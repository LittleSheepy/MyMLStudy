import tensorflow as tf
import numpy as np

print(tf.__version__)

print(np.histogram([1.1, 2, 1,1.5,4], 2, (0.0,1.0)))

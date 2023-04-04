from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy

import matplotlib.pyplot as plt

from skimage.segmentation import felzenszwalb
from skimage.data import coffee

img = coffee()
plt.imshow(img)
plt.show()
for colour_channel in (0, 1, 2):
    img[:, :, colour_channel] = skimage.feature.local_binary_pattern(
        img[:, :, colour_channel], 8,1.0,method='var')

plt.imshow(img)
plt.show()
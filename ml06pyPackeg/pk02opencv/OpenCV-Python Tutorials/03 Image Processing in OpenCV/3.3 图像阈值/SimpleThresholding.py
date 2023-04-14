import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('gradient.png', cv.IMREAD_GRAYSCALE)
ret,thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)         # binary
ret,thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)          # truncate 截断 大于thresh的都让等于thresh
ret,thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)         # 小于thresh的都=0
ret,thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)     # 大于thresh的都=0

titles = ['Original Image', 'BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3,thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])

plt.show()
import cv2

import numpy as np

from matplotlib import pyplot as plt

# pic_file = '../data/images/image_crocus_0003.png'

pic_file = r"../CB_00003.png"

img_bgr = cv2.imread(pic_file, cv2.IMREAD_COLOR)  # OpenCV读取颜色顺序：BGR
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

fig = plt.gcf()

fig.set_size_inches(5, 7.5)

plt.imshow(img_gray, cmap='gray')

plt.axis('off')

plt.title('Gray')

plt.show()
"""

cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) → hist

参数说明

images: 图片列表

channels: 需要计算直方图的通道。[0]表示计算通道0的直方图，[0,1,2]表示计算通道0,1,2所表示颜色的直方图

mask: 蒙版，只计算值>0的位置上像素的颜色直方图，取None表示无蒙版

histSize: 每个维度上直方图的大小，[8]表示把通道0的颜色取值等分为8份后计算直方图

ranges: 每个维度的取值范围，[lower0, upper0, lower1, upper1, ...]，lower可以取到，upper无法取到

hist: 保存结果的ndarray对象

accumulate: 是否累积，如果设置了这个值，hist不会被清零，直方图结果直接累积到hist中

"""

img_gray_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

plt.plot(img_gray_hist)

plt.title('Grayscale Histogram')

plt.xlabel('Bins')

plt.ylabel('# of Pixels')

plt.show()
















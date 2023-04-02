import cv2
import numpy as np
from matplotlib import pyplot as plt
# 读取图像
dir_root = r"D:\00myGitHub\opencv\samples\data/"
filename = dir_root + 'lena.jpg'
img = cv2.imread(filename)
# 计算B、G通道的直方图
hist = cv2.calcHist([img], [0,1], None, [256,256], [0,256,0,256])
# 显示B、G通道的直方图
plt.imshow(hist, interpolation='nearest')
plt.title('B,G Histogram')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.show()
import cv2 as cv
import numpy as np

img = cv.imread('j.png', cv.IMREAD_GRAYSCALE)
kernel = np.ones((5,5), np.uint8)
kernelRECT = cv.getStructuringElement(cv.MORPH_RECT ,(5,5))
kernelELLIPSE = cv.getStructuringElement(cv.MORPH_ELLIPSE ,(5,5))   # 椭圆
kernelCROSS = cv.getStructuringElement(cv.MORPH_CROSS ,(5,5))       # 十字
kernel = kernel

# 侵蚀
erosion = cv.erode(img, kernel, iterations=1)
# 扩张
dilation = cv.dilate(img, kernel, iterations=1)
# 开运算 先侵蚀后扩张
Opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# 闭运算 先扩张后侵蚀
Closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
# 形态梯度 扩张和腐蚀的差
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
gradient2 = cv.subtract(dilation, erosion)
# 礼帽 原图和开运算的差
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
# 黑帽 闭运算和原图的差
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)




cv.imshow('erode',np.hstack((img, erosion,dilation,Opening,Closing,gradient,gradient2,tophat,blackhat)))
cv.waitKey(0)
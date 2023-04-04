import cv2
import numpy as np

# 读取图像
dir_root = r"D:\02dataset\02opencv_data/"
filename = dir_root + 'hand0.jpg'
img = cv2.imread(filename)

# 将图像转换为HSV颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义直方图的参数
histSize = [16, 16]
ranges = [0, 180, 0, 256]
channels = [0, 1]

# 计算直方图
hist = cv2.calcHist([hsv], channels, None, histSize, ranges)

# 归一化直方图
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

# 计算反向投影图像
backProj = cv2.calcBackProject([hsv], channels, hist, ranges, 1)

# 对反向投影图像进行阈值处理
thresh = cv2.threshold(backProj, 200, 255, cv2.THRESH_BINARY_INV)[1]

# 在阈值处理后的图像中查找轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在轮廓周围绘制矩形框
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示带有矩形框的图像
cv2.imshow('backProj', backProj)
cv2.imshow('thresh', thresh)
cv2.imshow('Object Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

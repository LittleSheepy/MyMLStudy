"""
    计算图像的几何矩
"""
from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
import random as rng
from cv1findContours import my_findContours
rng.seed(12345)
"""
该函数的参数如下：
1. 输入图像：函数使用的输入图像。该图像必须为单通道8位或32位浮点类型。
2. 是否二值化：是否将输入图像进行二值化处理以获取更好的结果。该参数默认值为false。
3. 是否使用图像重心：是否使用图像重心作为几何矩的计算中心点。该参数默认值为false，即使用图像原点作为计算中心点。
4. 几何矩的阶数：指定要计算的几何矩的阶数。注意，计算的几何矩的阶数越高，运算时间会增加。参数默认值为0，表示计算所有的几何矩阶数。
返回值：函数返回一个Moments结构体，包含计算出的几何矩。该结构体包含以下成员：
m00: 零阶几何矩或区域面积。
m10: 一阶水平矩。
m01: 一阶垂直矩。
m20: 二阶水平矩。
m11: 二阶中心矩。
m02: 二阶垂直矩。
m30: 三阶水平矩。
m21: 三阶中心矩。
m12: 三阶垂直矩。
m03: 三阶主轴中心矩。
mu20: 二阶规范化水平矩。
mu11: 二阶规范化中心矩。
mu02: 二阶规范化垂直矩。
mu30: 三阶规范化水平矩。
mu21: 三阶规范化中心矩。
mu12: 三阶规范化垂直矩。
mu03: 三阶规范化主轴中心矩。
m10 / m00: 图像重心的水平位置。
m01 / m00: 图像重心的垂直位置。
"""
def my_moments(img_gray, threshold):
    contours, hierarchy, img_canny = my_findContours(img_gray, threshold)
    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv.moments(contours[i])
    return mu, contours

def thresh_callback(threshold):
    mu, contours = my_moments(img_gray, threshold)

    # 质心
    mc = [None]*len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

    # Draw contours
    ## [zeroMat]
    drawing = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    ## [zeroMat]
    ## [forContour]
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2)
        cv.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)
    cv.imshow('Contours', drawing)

    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for i in range(len(contours)):
        print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))
if __name__ == '__main__':

    dir_root = r"D:\02dataset\02opencv_data/"
    img_path = dir_root + 'hand1.jpg'
    img_bgr = cv.imread(cv.samples.findFile(img_path))
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    source_window = 'Source'
    cv.namedWindow(source_window)
    cv.imshow(source_window, img_bgr)
    max_thresh = 255
    thresh = 100 # initial threshold
    cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
    thresh_callback(thresh)

    cv.waitKey()

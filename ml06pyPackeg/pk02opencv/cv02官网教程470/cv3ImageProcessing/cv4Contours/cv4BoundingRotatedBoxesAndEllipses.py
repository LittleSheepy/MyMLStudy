"""
    创建边界旋转框和椭圆轮廓
"""
from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
from cv1findContours import my_findContours
rng.seed(12345)

# 最小外接矩形
def my_minAreaRect(img_gray, threshold):
    contours, hierarchy, img_canny = my_findContours(img_gray, threshold)
    minRect = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv.minAreaRect(c)
    return minRect, contours

# 拟合椭圆
def my_fitEllipse(img_gray, threshold):
    contours, hierarchy, img_canny = my_findContours(img_gray, threshold)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)
    return minEllipse, contours

def thresh_callback(threshold):
    minRect, contours = my_minAreaRect(img_gray, threshold)
    minEllipse, _ = my_fitEllipse(img_gray, threshold)

    drawing = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    # [画轮廓]
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color)        # 画轮廓
        if c.shape[0] > 5:
            cv.ellipse(drawing, minEllipse[i], color, 2)    # 画椭圆
        box = cv.boxPoints(minRect[i])
        box = np.intp(box)
        cv.drawContours(drawing, [box], 0, color)           # 画最小外接矩形
    cv.imshow('Contours', drawing)

if __name__ == '__main__':
    dir_root = r"D:\02dataset\02opencv_data/"
    img_path = dir_root + 'hand1.jpg'
    img_bgr = cv.imread(cv.samples.findFile(img_path))
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    source_window = 'Source'
    cv.namedWindow(source_window)
    cv.imshow(source_window, src)
    max_thresh = 255
    thresh = 100
    cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
    thresh_callback(thresh)

    cv.waitKey()

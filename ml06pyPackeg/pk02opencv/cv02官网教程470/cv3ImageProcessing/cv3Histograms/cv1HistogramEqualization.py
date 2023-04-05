"""
    直方图均衡化
"""
from __future__ import print_function
import cv2 as cv
import argparse

# 直方图均衡化
def my_equalizeHist():
    img_bgr = cv.imread(filename)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    img_hist = cv.equalizeHist(img_gray)
    return img_hist

# 分通道直方图均衡化
def myBGR_equalizeHist():
    img_bgr = cv.imread(filename)
    img_b = img_bgr[:, :, 0]
    img_g = img_bgr[:, :, 1]
    img_r = img_bgr[:, :, 2]
    img_b_hist = cv.equalizeHist(img_b)
    img_g_hist = cv.equalizeHist(img_g)
    img_r_hist = cv.equalizeHist(img_r)
    img_hist = cv.merge([img_b_hist, img_g_hist, img_r_hist])
    return img_hist


if __name__ == '__main__':
    dir_root = r"D:\00myGitHub\opencv\samples\data/"
    filename = dir_root + 'lena.jpg'
    img_bgr = cv.imread(filename)
    img_gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    img_hist = my_equalizeHist()
    img_bgr_hist = myBGR_equalizeHist()
    img_bgr_hist_gray = cv.cvtColor(img_bgr_hist, cv.COLOR_BGR2GRAY)
    img_show = cv.hconcat([img_gray, img_hist, img_bgr_hist_gray])
    cv.imshow('img_show', img_show)
    img_bgr_show = cv.hconcat([img_bgr, img_bgr_hist])
    cv.imshow('img_bgr_show', img_bgr_show)
    cv.waitKey()

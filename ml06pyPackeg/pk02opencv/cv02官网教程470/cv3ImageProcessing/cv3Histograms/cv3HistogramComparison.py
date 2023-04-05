from __future__ import print_function
from __future__ import division
import cv2
import cv2 as cv
import numpy as np
import argparse

def show_hist(histImage, hist1, hist2):
    cv.normalize(hist1, hist1, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(hist2, hist2, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, len(hist1)):
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(hist1[i - 1]))),
                (bin_w * (i), hist_h - int(np.round(hist1[i]))),
                (255, 0, 0), thickness=2)
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(hist2[i - 1]))),
                (bin_w * (i), hist_h - int(np.round(hist2[i]))),
                (255, 125, 0), thickness=2)

def my_compareHist(channels, histSize, ranges):
    src_base = cv.imread(filename0)
    src_test1 = cv.imread(filename1)
    hsv_base = cv.cvtColor(src_base, cv.COLOR_BGR2HSV)
    hsv_test1 = cv.cvtColor(src_test1, cv.COLOR_BGR2HSV)
    hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    hist_test1 = cv.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    """
    cv2.HISTCMP_CORREL：相关性方法，计算两个直方图之间的相关性。值越接近1，表示两个直方图越相似。
    cv2.HISTCMP_CHISQR：卡方方法，计算两个直方图之间的卡方距离。值越接近0，表示两个直方图越相似。
    cv2.HISTCMP_INTERSECT：交集方法，计算两个直方图之间的交集。值越大，表示两个直方图越相似。
    cv2.HISTCMP_BHATTACHARYYA：Bhattacharyya方法，计算两个直方图之间的Bhattacharyya距离。值越接近0，表示两个直方图越相似。
    cv2.HISTCMP_HELLINGER：Hellinger方法，计算两个直方图之间的Hellinger距离。值越接近0，表示两个直方图越相似。
    cv2.HISTCMP_CHISQR_ALT
    """
    base_test1 = cv.compareHist(hist_base, hist_test1, cv2.HISTCMP_BHATTACHARYYA)
    print(base_test1)
    return hist_base, hist_test1


#hsv_half_down = hsv_base[hsv_base.shape[0]//2:,:]
if __name__ == '__main__':
    dir_root = r"D:\02dataset\02opencv_data/"
    filename0 = dir_root + 'hand0.jpg'
    filename1 = dir_root + 'hand1.jpg'
    filename2 = dir_root + 'hand2.jpg'

    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / 256))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    src_base = cv.imread(filename0)
    src_test1 = cv.imread(filename1)

    hist_base2, hist_test2 = my_compareHist(channels=[0, 1], histSize=[256, 256], ranges=[0, 256, 0, 256])

    hist_base1, hist_test1 = my_compareHist(channels=[0], histSize=[255], ranges=[0, 256])
    show_hist(histImage, hist_base1, hist_test1)
    cv.imshow('hist_base2', hist_base2)
    cv.imshow('hist_test2', hist_test2)
    cv.imshow('calcHist Demo', histImage)
    cv.waitKey()


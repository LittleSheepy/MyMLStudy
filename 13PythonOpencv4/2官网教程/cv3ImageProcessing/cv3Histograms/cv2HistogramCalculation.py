from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse

def myGrayList_calcHist():
    bgr_planes = cv.split(img_bgr)  # [(512, 512),(512, 512),(512, 512)]
    b_hist = cv.calcHist(bgr_planes, [0], None, [256], (0, 256), accumulate=False)
    g_hist = cv.calcHist(bgr_planes, [1], None, [256], (0, 256), accumulate=False)
    r_hist = cv.calcHist(bgr_planes, [2], None, [256], (0, 256), accumulate=False)
    return b_hist, g_hist, r_hist

def mybgr_calcHist():
    bgr_planes = [img_bgr]  # [(512, 512, 3)]
    bg_hist = cv.calcHist(bgr_planes, [0,1], None, [256, 256], [0, 256, 0, 256])
    b_hist = cv.calcHist(bgr_planes, [0], None, [256], (0, 256), accumulate=False)
    g_hist = cv.calcHist(bgr_planes, [1], None, [256], (0, 256), accumulate=False)
    r_hist = cv.calcHist(bgr_planes, [2], None, [256], (0, 256), accumulate=False)
    return b_hist, g_hist, r_hist, bg_hist

def show_hist(histImage, hist, color=(255, 0, 0)):
    cv.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, len(hist)):
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(np.round(hist[i - 1]))),
                (bin_w * (i), hist_h - int(np.round(hist[i]))),
                color, thickness=2)

## [Display]
if __name__ == '__main__':
    dir_root = r"D:\00myGitHub\opencv\samples\data/"
    filename = dir_root + 'lena.jpg'
    img_bgr = cv.imread(filename)
    img_gray = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    b_hist, g_hist, r_hist = myGrayList_calcHist()
    b_hist_bgr, g_hist_bgr, r_hist_bgr, bg_hist_bgr = mybgr_calcHist()
    ## [Draw the histograms for B, G and R]
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / 256))

    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    histImage_bgr = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    ## [Draw the histograms for B, G and R]
    show_hist(histImage, b_hist, (255, 0, 0))
    show_hist(histImage, g_hist, (0, 255, 0))
    show_hist(histImage, r_hist, (0, 0, 255))

    show_hist(histImage_bgr, b_hist_bgr, (255, 0, 0))
    show_hist(histImage_bgr, g_hist_bgr, (0, 255, 0))
    show_hist(histImage_bgr, r_hist_bgr, (0, 0, 255))
    show_hist(histImage_bgr, bg_hist_bgr, (0, 0, 255))

    ## [Display]
    hist_show = cv.hconcat([histImage, histImage_bgr])
    cv.imshow('Source image', img_bgr)
    cv.imshow('calcHist Demo', hist_show)
    cv.waitKey()
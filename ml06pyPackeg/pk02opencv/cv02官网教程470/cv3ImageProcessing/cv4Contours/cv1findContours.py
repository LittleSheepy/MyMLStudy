from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)


# def findContours(image, mode, method, contours=None, hierarchy=None, offset=None):
# mode : 轮廓检索模式
#       cv.RETR_EXTERNAL    只检测外轮廓。
#       cv.RETR_LIST        检测所有轮廓，但不建立轮廓间的层次关系。
#       cv.RETR_CCOMP       检测所有轮廓，并将其组织为两级层次结构。顶层为连通域的外部边界，第二层为连通域的内部边界。
#       cv.RETR_TREE        检测所有轮廓，并重构嵌套轮廓的完整层次结构。
#       cv.RETR_FLOODFILL   image 要转 np.int32(image)，不然报错 np.int32(img_canny) img_canny
# method ：轮廓逼近方法，有以下几种可选方法：
#       cv2.CHAIN_APPROX_NONE：存储所有的轮廓点。
#       cv2.CHAIN_APPROX_SIMPLE：仅存储轮廓的端点。
#       cv2.CHAIN_APPROX_TC89_L1 和 cv2.CHAIN_APPROX_TC89_KCOS：使用 Teh - Chin 链逼近算法中的一种。
# 寻找轮廓
def my_findContours(img_gray, threshold):
    img_gray = cv.blur(img_gray, (3,3))
    img_canny = cv.Canny(img_gray, threshold, threshold * 2)
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy, img_canny

def thresh_callback(threshold):
    contours, hierarchy, img_canny = my_findContours(img_gray, threshold)

    drawing = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)

    # Show in a window
    img_canny = cv.cvtColor(img_canny, cv.COLOR_GRAY2BGR)
    cv.imshow(source_window, cv.hconcat([img_bgr, img_canny, drawing]))

if __name__ == '__main__':
    dir_root = r"D:\02dataset\02opencv_data/"
    img_path = dir_root + 'HappyFish.jpg'
    img_bgr = cv.imread(img_path)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # Create Window
    source_window = 'Source'
    cv.namedWindow(source_window)
    cv.imshow(source_window, img_bgr)
    cv.createTrackbar('Canny Thresh:', source_window, 100, 255, thresh_callback)
    thresh_callback(100)

    cv.waitKey()

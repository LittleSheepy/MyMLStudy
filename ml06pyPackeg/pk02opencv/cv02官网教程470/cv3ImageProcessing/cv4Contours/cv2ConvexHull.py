from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
from cv1findContours import my_findContours

rng.seed(12345)

# 凸包
def my_convexHull(img_gray, threshold):
    contours, hierarchy, img_canny = my_findContours(img_gray, threshold)
    hull_list = []
    for i in range(len(contours)):
        hull = cv.convexHull(contours[i])
        hull_list.append(hull)
    return hull_list, contours, img_canny

def thresh_callback(threshold):
    hull_list, contours, img_canny = my_convexHull(img_gray, threshold)
    # Draw contours + hull results
    drawing = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color)
        cv.drawContours(drawing, hull_list, i, color)

    img_canny = cv.cvtColor(img_canny, cv.COLOR_GRAY2BGR)
    cv.imshow(source_window, cv.hconcat([img_bgr, img_canny, drawing]))

if __name__ == '__main__':
    dir_root = r"D:\02dataset\02opencv_data/"
    img_path = dir_root + 'hand1.jpg'
    img_bgr = cv.imread(img_path)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

    # Create Window
    source_window = 'Source'
    cv.namedWindow(source_window)
    cv.imshow(source_window, img_bgr)
    max_thresh = 255
    thresh = 100 # initial threshold
    cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
    thresh_callback(thresh)

    cv.waitKey()

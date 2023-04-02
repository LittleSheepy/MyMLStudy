"""
    侵蚀和膨胀
"""
import cv2
import cv2 as cv
import numpy as np
import random


def erosion(val):
    erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_window)
    erosion_type = 0
    val_type = cv.getTrackbarPos(title_trackbar_element_type, title_window)
    if val_type == 0:
        erosion_type = cv.MORPH_RECT
    elif val_type == 1:
        erosion_type = cv.MORPH_CROSS
    elif val_type == 2:
        erosion_type = cv.MORPH_ELLIPSE

    element = cv.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    erosion_dst = cv.erode(src, element)
    erosion_dst = cv.dilate(erosion_dst, element)
    cv.imshow(title_erosion_window, erosion_dst)

def dilatation(val):
    dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_window)
    dilatation_type = 0
    val_type = cv.getTrackbarPos(title_trackbar_element_type, title_window)
    if val_type == 0:
        dilatation_type = cv.MORPH_RECT
    elif val_type == 1:
        dilatation_type = cv.MORPH_CROSS
    elif val_type == 2:
        dilatation_type = cv.MORPH_ELLIPSE

    element = cv.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src, element)
    cv.imshow(title_dilatation_window, dilatation_dst)
def main():
    cv.namedWindow(title_window)
    cv.createTrackbar(title_trackbar_element_type, title_window, 0, 2, erosion)
    cv.createTrackbar(title_trackbar_kernel_size, title_window, 0, 21, erosion)
    erosion(0)
    cv.waitKey()

if __name__ == "__main__":
    erosion_size = 0
    max_elem = 2
    max_kernel_size = 10
    title_trackbar_element_type = '0: Rect;1: Cross;2: Ellipse'
    title_trackbar_kernel_size = 'Kernel size: 2n +1'
    dir_root = r"D:\00myGitHub\opencv\samples\data/"
    filename = dir_root + 'LinuxLogo.jpg'
    src = cv.imread(cv.samples.findFile(filename))
    title_window = "erosion_dilatation"
    title_erosion_window = title_window
    title_dilatation_window = title_window
    main()


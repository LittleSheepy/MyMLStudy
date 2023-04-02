"""
    开闭和礼帽运算
"""
import cv2
import cv2 as cv
import numpy as np
import random

# 开运算 先侵蚀后膨胀
def my_opening():
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3), (1, 1))
    img_opening = cv.morphologyEx(img_bgr, cv.MORPH_OPEN, element)
    img_show = cv2.hconcat([img_bgr, img_opening])
    cv.imshow("img_opening", img_show)
    cv.waitKey()

# 闭运算 先膨胀后侵蚀
def my_closing():
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3), (1, 1))
    img_closing = cv.morphologyEx(img_bgr, cv.MORPH_CLOSE, element)
    img_show = cv2.hconcat([img_bgr, img_closing])
    cv.imshow("img_closing", img_show)
    cv.waitKey()

# 梯度 轮廓
def my_gradient():
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3), (1, 1))
    img_gradient = cv.morphologyEx(img_bgr, cv.MORPH_GRADIENT, element)
    img_show = cv2.hconcat([img_bgr, img_gradient])
    cv.imshow("img_gradient", img_show)
    cv.waitKey()

# input - opening
def my_tophat():
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3), (1, 1))
    img_tophat = cv.morphologyEx(img_bgr, cv.MORPH_TOPHAT, element)
    img_show = cv2.hconcat([img_bgr, img_tophat])
    cv.imshow("img_tophat", img_show)
    cv.waitKey()

# input - closing
def my_blackhat():
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3), (1, 1))
    img_blackhat = cv.morphologyEx(img_bgr, cv.MORPH_BLACKHAT, element)
    img_show = cv2.hconcat([img_bgr, img_blackhat])
    cv.imshow("img_tophat", img_show)
    cv.waitKey()


def morphology_operations(val):
    morph_operator = cv.getTrackbarPos(title_trackbar_operator_type, title_window)
    morph_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_window)
    morph_elem = 0
    val_type = cv.getTrackbarPos(title_trackbar_element_type, title_window)
    if val_type == 0:
        morph_elem = cv.MORPH_RECT
    elif val_type == 1:
        morph_elem = cv.MORPH_CROSS
    elif val_type == 2:
        morph_elem = cv.MORPH_ELLIPSE

    element = cv.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
    operation = morph_op_dic[morph_operator]
    dst = cv.morphologyEx(src, operation, element)
    cv.imshow(title_window, dst)

def main():
    cv.namedWindow(title_window)
    cv.createTrackbar(title_trackbar_operator_type, title_window, 0, max_operator, morphology_operations)
    cv.createTrackbar(title_trackbar_element_type, title_window, 0, max_elem, morphology_operations)
    cv.createTrackbar(title_trackbar_kernel_size, title_window, 0, max_kernel_size, morphology_operations)

    morphology_operations(0)
    cv.waitKey()

if __name__ == "__main__":
    morph_size = 0
    max_operator = 4
    max_elem = 2
    max_kernel_size = 21
    title_trackbar_operator_type = 'Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat'
    title_trackbar_element_type = 'Element:\n 0: Rect - 1: Cross - 2: Ellipse'
    title_trackbar_kernel_size = 'Kernel size:\n 2n + 1'
    title_window = 'Morphology Transformations Demo'
    morph_op_dic = {0: cv.MORPH_OPEN, 1: cv.MORPH_CLOSE, 2: cv.MORPH_GRADIENT, 3: cv.MORPH_TOPHAT, 4: cv.MORPH_BLACKHAT}

    dir_root = r"D:\00myGitHub\opencv\samples\data/"
    filename = dir_root + 'LinuxLogo.jpg'
    img_bgr = cv.imread(filename)
    src = img_bgr.copy()
    my_opening()


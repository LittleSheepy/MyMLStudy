"""
    平滑图像
"""
import cv2
import cv2 as cv
import numpy as np
import random
# 使用归一化框过滤器模糊图像。
def my_blur():
    img_bgr = cv.imread(cv.samples.findFile(filename))  # 读图
    for i in range(1, 32, 2):
        img_blur = cv.blur(img_bgr, (i, i))
        cv.imshow("img_blur", img_blur)
        cv.waitKey(1000)

def my_GaussianBlur():
    img_bgr = cv.imread(cv.samples.findFile(filename))  # 读图
    for i in range(1, 32, 2):
        img_GaussianBlur = cv.GaussianBlur(img_bgr, (i, i), 0)
        cv.imshow("img_GaussianBlur", img_GaussianBlur)
        cv.waitKey(1000)

def my_medianBlur():
    img_bgr = cv.imread(cv.samples.findFile(filename))  # 读图
    for i in range(1, 32, 2):
        img_medianBlur = cv.medianBlur(img_bgr, i)
        cv.imshow("img_medianBlur", img_medianBlur)
        cv.waitKey(1000)

if __name__ == "__main__":
    dir_root = r"D:\00myGitHub\opencv\samples\data/"
    filename = dir_root + 'lena.jpg'
    my_medianBlur()

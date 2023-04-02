"""
    图像金字塔
"""
import numpy as np
import sys
import cv2 as cv

def my_pyrUp(img):
    rows, cols, _channels = map(int, img.shape)
    # pyrUp(src, dst=None, dstsize=None, borderType=None)
    img = cv.pyrUp(img, dstsize=(2 * cols, 2 * rows), borderType=cv.BORDER_DEFAULT)
    return img

def my_pyrDown(img):
    rows, cols, _channels = map(int, img.shape)
    img = cv.pyrDown(img, dstsize=(cols // 2, rows // 2), borderType=cv.BORDER_DEFAULT)
    return img

def main():
    img_bgr = cv.imread(cv.samples.findFile(filename))
    while 1:
        cv.imshow('Pyramids Demo', img_bgr)
        k = cv.waitKey(0)
        if k == 27: break
        elif chr(k) == 'i':
            img_bgr = my_pyrUp(img_bgr)
        elif chr(k) == 'o':
            img_bgr = my_pyrDown(img_bgr)
    cv.destroyAllWindows()
    return 0

if __name__ == "__main__":
    dir_root = r"D:\00myGitHub\opencv\samples\data/"
    filename = dir_root + 'sudoku.png'
    main()

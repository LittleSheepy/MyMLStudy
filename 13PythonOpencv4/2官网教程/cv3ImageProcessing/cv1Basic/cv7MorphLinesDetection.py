"""
    形态转换检测线
"""
import numpy as np
import sys
import cv2 as cv


def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def get_bin_img():
    dir_root = r"D:\00myGitHub\opencv\doc\tutorials\imgproc\morph_lines_detection\images/"
    filename = dir_root + 'src.png'
    src = cv.imread(filename, cv.IMREAD_COLOR)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    bitwise_not = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(bitwise_not, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    cv.imshow("binary", bw)
    return bw

def get_horiz_line():
    horizontal = get_bin_img()
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (50, 1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    cv.imshow("horizontal", horizontal)

def get_vert_line():
    vertical = get_bin_img()
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, 4))
    vertical = cv.erode(vertical, horizontalStructure)
    vertical = cv.dilate(vertical, horizontalStructure)
    cv.imshow("vertical", vertical)
    return vertical

def main():
    get_horiz_line()
    vertical = get_vert_line()
    vertical_bit = cv.bitwise_not(vertical)
    cv.imshow("vertical_bit", vertical_bit)

    '''
    Extract edges and smooth image according to the logic 根据逻辑提取边缘、平滑图像
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''

    # Step 1
    edges = cv.adaptiveThreshold(vertical_bit, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 0)
    cv.imshow("edges", edges)

    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges_dilate = cv.dilate(edges, kernel)
    cv.imshow("edges_dilate", edges_dilate)

    # Step 3
    smooth = np.copy(vertical_bit)
    cv.imshow("smooth", smooth)

    # Step 4
    smooth = cv.blur(smooth, (2, 2))
    cv.imshow("smooth_blur", smooth)

    # Step 5
    (rows, cols) = np.where(edges_dilate != 0)
    vertical_bit[rows, cols] = smooth[rows, cols]

    # Show final result
    cv.imshow("smooth - final", vertical_bit)
    # [smooth]

    cv.waitKey(0)
    return 0

if __name__ == "__main__":
    main()

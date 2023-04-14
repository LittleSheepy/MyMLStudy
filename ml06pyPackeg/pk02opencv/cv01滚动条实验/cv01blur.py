import sys
import cv2
import numpy as np


def Bar1Call(val):
    global img_gray
    times = cv2.getTrackbarPos('times', window_name)
    para1 = cv2.getTrackbarPos('para1', window_name)
    para2 = cv2.getTrackbarPos('para2', window_name)
    para3 = cv2.getTrackbarPos('para3', window_name)
    img_gray = cv2.blur(img_gray, (3, 3))
    if para1 == 0:
        para1 = 1
    img_blur = img_gray.copy()
    for i in range(times):
        img_blur = cv2.blur(img_blur, (para1, para1))
    img_del = abs(img_gray.astype(np.float64) - img_blur.astype(np.float64))
    img_del_max = img_del.max()
    img_del = img_del/img_del_max*255
    img_del = img_del.astype(np.uint8)
    for i in range(para3):
        img_del = cv2.blur(img_del, (5, 5))
    ret, thresh = cv2.threshold(img_del, para2, 255, cv2.THRESH_BINARY)

    cv2.imshow(window_name, cv2.hconcat([img_gray, img_del, thresh]))


def creatWindow():
    cv2.namedWindow(window_name)
    cv2.createTrackbar('times', window_name, 0, 100, Bar1Call)
    cv2.createTrackbar('para2', window_name, 0, 255, Bar1Call)
    cv2.createTrackbar('para3', window_name, 0, 10, Bar1Call)
    cv2.createTrackbar('para1', window_name, 500, 600, Bar1Call)
    Bar1Call(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dir_root = r"D:\04DataSets\ningjingLG\02ZangWu\/"
    filename = dir_root + 'gray_0045737_CM2_2_box.bmp'
    img_bgr = cv2.imread(filename)
    img_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    window_name = "BlurWindow"
    creatWindow()

""" 
1 500 40


"""

import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random as rng
#
# class CNumRec:
#     def __init__(self):

if __name__ == '__main__':
    dir_root = r"D:\03GitHub\00myGitHub\MyMLStudy\ml00project\pj2LG\numRec/"
    img_path = dir_root + "gray_0002314_CM1_1.bmp"
    white_template_path = dir_root + "white_template3.bmp"
    # 文字位置
    nums_pos = []
    img_gray = cv.imread(img_path, cv2.IMREAD_GRAYSCALE)
    id_card = cv.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_white_gray = cv2.imread(white_template_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(img_gray, img_white_gray, cv2.TM_CCOEFF_NORMED)
    max_ = res.max()
    # 画出匹配结果
    threshold = max_
    loc = np.where(res >= threshold)
    xmin = loc[1][0]
    ymin = loc[0][0]
    h, w = img_white_gray.shape
    img_cut = img_gray[ymin:ymin+h, xmin:xmin+w]

    ret, img_cut_binary_img = cv.threshold(img_cut, 250, 255, cv.THRESH_BINARY_INV)
    # 定义水平膨胀核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    # 水平膨胀
    img_closing = cv.morphologyEx(img_cut_binary_img, cv.MORPH_CLOSE, kernel)
    dilated_img = cv.dilate(img_closing, kernel3, iterations=1)
    # 找轮廓
    contours, hierarchy = cv.findContours(dilated_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    # 按照特征找
    result_rect = [0, 1000, 0, 0]
    for rect in boundRect:
        if rect[2]/rect[3] > 5.0 and result_rect[1] > rect[1] > 200:
            result_rect = rect

    binary_img = img_cut_binary_img[result_rect[1]:result_rect[1]+result_rect[3], result_rect[0]:result_rect[0]+result_rect[2]]

    cv.imshow('Binary image', binary_img)
    # 模板
    template_list = []
    for i in range(10):
        img_i_path = dir_root + str(i) + ".bmp"
        img_i = cv.imread(img_i_path, cv2.IMREAD_GRAYSCALE)
        _, binary_img_i = cv.threshold(img_i, 250, 255, cv.THRESH_BINARY_INV)
        template_list.append(binary_img_i)

    result = []     # class, xmin, ymin
    xmin  = 0
    while True:
        img_tmp = binary_img[:, xmin:xmin+40]
        max_score = 0
        res_loc = []
        xmin_ = 0
        for i, template_img in enumerate(template_list):
            res = cv2.matchTemplate(img_tmp, template_img, cv2.TM_CCOEFF_NORMED)
            max_ = res.max()
            if max_ > max_score:
                threshold = max_
                loc = np.where(res >= threshold)
                xmin_ = loc[1][0]
                ymin_ = loc[0][0]
                max_score = max_
                max_index = i
                res_loc = [i, xmin + xmin_, ymin_]
        result.append(res_loc)
        xmin = xmin + xmin_ + 25
        if xmin+40 > binary_img.shape[1] or len(result) >= 7:
            break

    print(result)

    cv.waitKey(0)

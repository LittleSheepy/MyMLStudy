import os
import cv2
import cv2 as cv
import numpy as np
import random as rng

def verticalLine(gray):
    lineList = []
    # 应用高斯模糊以减少噪音
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    blur = cv2.GaussianBlur(blur, (7, 7), 0)
    cv2.imwrite(out_dir + "01blur.jpg", blur)
    # 检测边缘
    edges = cv2.Canny(blur, 30, 90, apertureSize=3)
    cv2.imwrite(out_dir + "01edges.jpg", edges)
    # 应用霍夫直线变换以检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=40)
    # 绘制检测到的直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) * 10 < abs(y1 - y2):
            lineList.append(line[0])
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return lineList


def lineGroup(line_list):
    lineList_sort = sorted(line_list, key=lambda x: x[0])
    lines_group = [[]]
    lines = []
    last_x = 0
    for line in lineList_sort:
        if len(lines_group[-1]) == 0:
            lines_group[-1].append(line)
        elif line[0] - last_x < 50:
            lines_group[-1].append(line)
        else:
            lines_group.append([line])
        last_x = line[0]
    lines.append(line)
    return lines_group


# 筛选 计算中心点
def getLineCenterX(lines_group, top_y, top_x=0):
    global centerList
    # 帅选
    # 去掉白色区域 计算中心点
    lines_group_new = []
    center_xs = []
    dis_max_index = 0
    dis_max = 0
    for lines in lines_group:
        xs = [line[0] for line in lines] + [line[2] for line in lines]
        ys = [line[1] for line in lines] + [line[3] for line in lines]
        xs_min, xs_max, ys_min, ys_max = min(xs), max(xs), min(ys), max(ys)
        if top_x > 0:
            if xs_min < top_x+10 and top_x-750 < xs_max:
                continue
        if ys_min < (top_y + 300) or ys_max > top_y + 1100:
            lines_group_new.append(lines)
            # 中心
            center_xs.append(int((xs_min + xs_max) / 2))
            if len(center_xs) > 1:
                dis = center_xs[-1] - center_xs[-2]
                if dis > dis_max:
                    dis_max = dis
                    dis_max_index = len(center_xs) - 1
    centerList.append(center_xs[dis_max_index])
    return lines_group_new, center_xs, dis_max_index


# 编号 img1 ： 0 12  3
# 编号 img4 ： 8  9A B
def BianHao14(img_gray, img_serial=1):
    keySerialDict = {"1": 3, "4": 9}
    key_serial = keySerialDict[str(img_serial)]
    top_y = getTop(img_gray)
    lineList = verticalLine(img_gray)

    # 分组
    linesAll = lineGroup(lineList)

    # 筛选 计算中心点
    linesAll_new, center_xs, dis_max_index = getLineCenterX(linesAll, top_y)

    # 保存结果
    resultDict = {}
    for i in range(len(center_xs)):
        line_serial = key_serial + i - dis_max_index
        key_name = str(line_serial)
        result = {"center_x": center_xs[i], "lines": linesAll_new[i]}
        resultDict[key_name] = result

    resultDict["linesAll"] = lineList
    resultDict["top_y"] = top_y
    resultDict["serial"] = img_serial
    return resultDict

# 编号 img1 ： 0 12  3
# 编号 img4 ： 8  9A B
def BianHao1(img_gray, img_serial=1):
    keySerialDict = {"1": 3, "4": 9}
    key_serial = keySerialDict[str(img_serial)]
    top_y = getTop(img_gray)
    top_x = getright(img_gray)
    lineList = verticalLine(img_gray)

    # 分组
    linesAll = lineGroup(lineList)

    # 筛选 计算中心点
    linesAll_new, center_xs, dis_max_index = getLineCenterX(linesAll, top_y, top_x)

    # 保存结果
    resultDict = {}
    for i in range(len(center_xs)):
        line_serial = key_serial + i - dis_max_index
        key_name = str(line_serial)
        result = {"center_x": center_xs[i], "lines": linesAll_new[i]}
        resultDict[key_name] = result

    resultDict["linesAll"] = lineList
    resultDict["top_y"] = top_y
    resultDict["serial"] = img_serial
    return resultDict


# 编号 img2 ： |3   4   5  |
# 编号 img3 ： |  6    7   8|
def BianHao23(img_gray, img_serial=2):
    keySerialDict = {"2": 5, "3": 7}
    key_serial = keySerialDict[str(img_serial)]

    # 顶边
    top_y = getTop(img_gray)
    lineList = verticalLine(img_gray)

    # 分组
    linesAll = lineGroup(lineList)

    # 筛选 计算中心点
    linesAll_new, center_xs, dis_max_index = getLineCenterX(linesAll, top_y)

    # 保存结果
    resultDict = {}
    for i in range(len(center_xs)):
        line_serial = key_serial + i - dis_max_index
        key_name = str(line_serial)
        result = {"center_x": center_xs[i], "lines": linesAll_new[i]}
        resultDict[key_name] = result

    resultDict["linesAll"] = lineList
    resultDict["top_y"] = top_y
    resultDict["serial"] = img_serial
    return resultDict


def BianHao(img_gray, img_serial=2):
    func_list = [BianHao1, BianHao23, BianHao23, BianHao14]
    return func_list[int(img_serial) - 1](img_gray, img_serial)


def getTop(img_gray):
    h, w = img_gray.shape
    point_x = int(w / 2)
    for point_y in range(h):
        if img_gray[point_y, point_x] > 50:
            return point_y

def getright(img_gray):
    h, w = img_gray.shape
    point_y = 1200
    for point_x in range(w-1, int(h/2), -1):
        if img_gray[point_y, point_x] > 240:
            wList.append(point_x)
            return point_x


def find_white_region(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用二值化提取白色区域
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # 寻找轮廓并筛选白色区域
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_contours = []
    for cnt in contours:
        if 320000 > cv2.contourArea(cnt) > 280000:
            x, y, w, h = cv2.boundingRect(cnt)
            white_contours.append((x, y, w, h))
    # 绘制所有
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(image, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # 返回最大的白色区域的外界矩形
    if white_contours:
        x, y, w, h = max(white_contours, key=lambda x: x[2] * x[3])

        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        return (x, y, w, h), thresh
    else:
        return None

def save_white_region():
    tested = []
    for file_name in os.listdir(img_dir):
        first_name = file_name[:-5]
        if first_name in tested:
            continue
        print(first_name)
        tested.append(first_name)
        img_bgr = cv2.imread(img_dir + first_name + "1.bmp")
        (x, y, w, h), thresh = find_white_region(img_bgr)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(out_dir + first_name[:-1] + ".jpg", img_bgr)
        cv2.imwrite(out_dir + first_name[:-1] + "_thr.jpg", thresh)
# 模板匹配
def my_matchTemplate(img_bgr, tmpl_bgr, method=cv.TM_SQDIFF_NORMED, mask=None):
    # def matchTemplate(image, templ, method, result=None, mask=None)
    #
    # cv2.TM_SQDIFF         采用<平方差>的方法，求模板和图像之间的差值，差值越小匹配度越高。
    # cv2.TM_SQDIFF_NORMED  采用归一化平方差的方法，将模板和图像归一化后再求平方差，差值越小匹配度越高。
    # cv2.TM_CCORR          采用<互相关>的方法，将模板和图像卷积后再求相关，相关值越大匹配度越高。
    # cv2.TM_CCORR_NORMED
    # cv2.TM_CCOEF          采用<相关系数>的方法，将模板和图像转化为概率分布后再求相关系数，相关系数越大匹配度越高。
    # cv2.TM_CCOEFF_NORMED
    result = cv.matchTemplate(img_bgr, tmpl_bgr, method, result=None, mask=mask)
    cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
    if method == cv.TM_SQDIFF or method == cv.TM_SQDIFF_NORMED:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    return matchLoc, result

def drawResultDict(img, result_dict):
    #
    linesAll = result_dict.get("linesAll")
    for line in linesAll:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)
    for key in result_dict:
        if isinstance(result_dict[key], dict):
            lines = result_dict[key].get("lines")
            if not lines:
                continue
            for line in lines:
                cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)
    return img

def test(img_dir, out_dir):
    tested = []
    for file_name in os.listdir(img_dir):
        first_name = file_name[:-5]
        if first_name in tested:
            continue
        print(first_name)
        tested.append(first_name)
        img_list = []
        resultDictList = []
        for i in range(4):
            img_bgr = cv2.imread(img_dir + first_name + str(i + 1) + ".bmp")
            img_gray = cv2.imread(img_dir + first_name + str(i + 1) + ".bmp", cv2.IMREAD_GRAYSCALE)
            result_dict = BianHao(img_gray, i + 1)
            # resultDictList.append(result_dict)
            drawResultDict(img_bgr, result_dict)
            img_list.append(img_bgr)
        img = np.concatenate((img_list[0], img_list[1]), axis=1)
        img = np.concatenate((img, img_list[2]), axis=1)
        img = np.concatenate((img, img_list[3]), axis=1)
        cv2.imwrite(out_dir + first_name[:-1] + ".jpg", img)
centerList = []
wList = []
if __name__ == '__main__':
    dir_root = r"D:\04DataSets\ningjingLG/"
    img_dir = dir_root + "all/"
    out_dir = dir_root + "out4/"
    img1 = cv2.imread(dir_root + r'\black\black_R032829_CM1_1.bmp')
    img2 = cv2.imread(dir_root + r'\black\black_R032829_CM1_2.bmp')
    template2 = cv2.imread(dir_root + r"template2.bmp")
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # #BianHao1(img_gray, img_serial)
    # top_y = getTop(img_gray)
    # print("top_y", top_y)
    # result_dict = BianHao1(img_gray)
    # drawResultDict(img1, result_dict)
    # cv2.imwrite(out_dir + "01.jpg", img1)
    # test(img_dir, out_dir)
    # wList = np.array(wList)
    #save_white_region()
    find_white_region(img1)
    cv2.imwrite(dir_root + "region.jpg", img1)
    matchLoc, result = my_matchTemplate(img1, template2, cv2.TM_CCOEFF_NORMED)


    pass

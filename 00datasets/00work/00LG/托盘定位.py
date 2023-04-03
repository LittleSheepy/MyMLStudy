import os
import cv2
import numpy as np


def verticalLine(gray):
    lineList = []
    # 应用高斯模糊以减少噪音
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 检测边缘
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    # 应用霍夫直线变换以检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
    # 绘制检测到的直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) * 10 < abs(y1 - y2):
            lineList.append(line[0])
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return lineList


def lineGroup(line_list):
    lineList_sort = sorted(line_list, key=lambda x: x[0])
    lines_group = []
    lines = []
    last_x = 0
    for line in lineList_sort:
        if len(lines) == 0:
            lines.append(line)
            last_x = line[0]
            continue
        elif line[0] - last_x < 50:
            lines.append(line)
        else:
            lines_group.append(lines)
            lines = []
    return lines_group


# 筛选 计算中心点
def getLineCenterX(lines_group, top_y):
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
        if ys_min < (top_y + 400) or ys_max > top_y + 1000:
            lines_group_new.append(lines)
            # 中心
            center_xs.append(int((xs_min + xs_max) / 2))
            if len(center_xs) > 1:
                dis = center_xs[-1] - center_xs[-2]
                if dis > dis_max:
                    dis_max = dis
                    dis_max_index = len(center_xs) - 1
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
    func_list = [BianHao14, BianHao23, BianHao23, BianHao14]
    return func_list[int(img_serial) - 1](img_gray, img_serial)


def getTop(img_gray):
    h, w = img_gray.shape
    point_x = int(w / 2)
    for point_y in range(h):
        if img_gray[point_y, point_x] > 50:
            return point_y


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

if __name__ == '__main__':
    dir_root = r"D:\04DataSets\ningjingLG/"
    img_dir = dir_root + "all/"
    out_dir = dir_root + "out2/"
    img1 = cv2.imread(dir_root + r'\black\black_0074690_CM1_1.bmp')
    img2 = cv2.imread(dir_root + r'\black\black_0074690_CM1_2.bmp')
    # img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # #BianHao1(img_gray, img_serial)
    # top_y = getTop(img_gray)
    # print("top_y", top_y)
    # BianHao1(img_gray)
    test(img_dir, out_dir)

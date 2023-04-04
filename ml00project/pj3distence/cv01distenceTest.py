import math
import cv2
import numpy as np
import time


# 耳朵
def getEarPoint(img_gray, point_x, point_y_start=0):
    h = img_gray.shape[0]
    y = point_y_start
    EarPoints = {}
    for point_y in range(h):
        if img_gray[point_y, point_x] > 50:
            y = point_y
            EarPoints["top"] = [point_x, point_y]
            break
    for point_y in range(y+1, h, 1):
        if img_gray[point_y, point_x] < 50:
            y = point_y
            EarPoints["bottom"] = [point_x, point_y]
            break
    return EarPoints

# 遍历列
def getColumnPoint(img_gray, point_x, point_y_start=0):
    h = img_gray.shape[0]
    y = point_y_start
    ColumnPoints = {}
    for point_y in range(h):
        if img_gray[point_y, point_x] > 200:
            y = point_y
            ColumnPoints["whitetop"] = [point_x, point_y]
            break
    for point_y in range(y+1, h, 1):
        if img_gray[point_y, point_x] < 100:
            y = h
            ColumnPoints["whitebottom"] = [point_x, point_y]
            break
        if img_gray[point_y, point_x] < 160:
            y = point_y
            ColumnPoints["blacktop"] = [point_x, point_y]
            break
    for point_y in range(y+1, h, 1):
        if img_gray[point_y, point_x] > 200:
            y = point_y
            ColumnPoints["blackbottom"] = [point_x, point_y]
            break
    for point_y in range(y+1, h, 1):
        if img_gray[point_y, point_x] < 100:
            y = point_y
            ColumnPoints["whitebottom"] = [point_x, point_y]
            break
    return ColumnPoints

# 遍历行
def getRowPoint(img_gray, point_x, point_y_start=0):
    h = img_gray.shape[0]
    y = point_y_start
    ColumnPoints = {}
    for point_y in range(h):
        if img_gray[point_y, point_x] > 200:
            y = point_y
            ColumnPoints["whitetop"] = [point_x, point_y]
            break
    for point_y in range(y+1, h, 1):
        if img_gray[point_y, point_x] < 100:
            y = h
            ColumnPoints["whitebottom"] = [point_x, point_y]
            break
        if img_gray[point_y, point_x] < 160:
            y = point_y
            ColumnPoints["blacktop"] = [point_x, point_y]
            break
    for point_y in range(y+1, h, 1):
        if img_gray[point_y, point_x] > 200:
            y = point_y
            ColumnPoints["blackbottom"] = [point_x, point_y]
            break
    for point_y in range(y+1, h, 1):
        if img_gray[point_y, point_x] < 100:
            y = point_y
            ColumnPoints["whitebottom"] = [point_x, point_y]
            break
    return ColumnPoints

def getRowWhitePoint(img_gray, point_y):
    w = img_gray.shape[1]
    RowPoints = {}
    for point_x in range(w):
        if img_gray[point_y, point_x] > 200:
            RowPoints["whiteleft"] = [point_x, point_y]
            break
    for point_x in range(w-1, 0, -1):
        if img_gray[point_y, point_x] > 200:
            RowPoints["whiteright"] = [point_x, point_y]
            break
    return RowPoints


def line_angle(line1, line2):
    point1 = line1[0]
    point2 = line1[1]
    point3 = line2[0]
    point4 = line2[1]
    # 计算两条直线的斜率
    slope1 = (point2[1] - point1[1]) / (point2[0] - point1[0]+0.0001)
    #slope2 = (point4[1] - point3[1]) / (point4[0] - point3[0]+0.0001)
    slope2 = 0
    # 计算两条直线的夹角
    angle = math.atan(abs((slope1 - slope2) / (1 + slope1 * slope2)))
    angle = math.degrees(angle)
    # 计算两条直线的交点
    x = (point1[1] - point3[1] + slope2 * point3[0] - slope1 * point1[0]) / (slope2 - slope1)
    y = slope1 * (x - point1[0]) + point1[1]
    # 输出结果
    print("两条直线的夹角为：%.2f度" % angle)
    print("两条直线的交点为：(%.2f, %.2f)" % (x, y))
    return angle, [int(x), int(y)]

def line_angle_out(outline1, outline2):
    # 计算两条直线的斜率
    k1 = outline1[1] / (outline1[0]+0.000001)
    b1 = outline1[3] - k1 * outline1[2]
    k2 = outline2[1] / (outline2[0]+0.000001)
    b2 = outline2[3] - k2 * outline2[2]
    # 计算两条直线的夹角
    angle = math.atan(abs((k1 - k2) / (1 + k1 * k2)))
    angle = math.degrees(angle)
    # 计算两条直线的交点
    x = (b1 - b2) / (k2 - k1)
    y = k1 * x + b1
    # 输出结果
    print("两条直线的夹角为：%.2f度" % angle)
    print("两条直线的交点为：(%.2f, %.2f)" % (x, y))
    return angle, [int(x), int(y)]

def fitLine(PointsDictList, lineType):
    print("fitLine >> ", lineType)
    resultDict = {}
    PointsList = []
    for ColumnPoints in PointsDictList:
        point = ColumnPoints.get(lineType, [])
        if point:
            PointsList.append(ColumnPoints[lineType])
    PointsList = np.array(PointsList)
    resultDict["PointsListOri"] = PointsList.copy()
    Line = cv2.fitLine(PointsList, cv2.DIST_L2, 0, 0.01, 0.01)
    Line = Line.reshape(4)
    for i in range(10):
        # 计算距离
        distances = []
        count = int(len(PointsList)/10)+1
        if Line[0] == 0:
            for point in PointsList:
                dis = abs(point[0] - Line[2])
                distances.append(dis)
        else:
            for point in PointsList:
                k = Line[1] / Line[0]
                b = Line[3] - k * Line[2]
                dis = abs(k * point[0] - point[1] + b) / np.sqrt(k ** 2 + 1)
                distances.append(dis)

        try:
            max_dis = max(distances)
        except Exception as e:
            print(e)
        print("max_dis ", max_dis)
        if max_dis < 5:
            print("拟合结束 ", lineType, i)
            break
        sorted_indexes = np.argsort(distances)[:-count]
        PointsList = np.array(PointsList)[sorted_indexes]
        Line = cv2.fitLine(PointsList, cv2.DIST_L2, 0, 0.01, 0.01)
        Line = Line.reshape(4)

    resultDict["PointsList"] = PointsList
    resultDict["Line"] = Line.reshape(4)
    return resultDict

def distenceMeasure(img_gray):
    resultDict = {}
    # 计算 关键点
    ColumnKeyPoints = getColumnPoint(img_gray, 220)     # 220
    resultDict["ColumnKeyPoints"] = ColumnKeyPoints
    RowPoints_y = ColumnKeyPoints["whitetop"][1] + int((ColumnKeyPoints["blacktop"][1] - ColumnKeyPoints["whitetop"][1])/2)
    RowKeyPoints = getRowWhitePoint(img_gray, RowPoints_y)
    resultDict["RowKeyPoints"] = RowKeyPoints

    # ....... Column 获取点列表
    ColumnPointsDictList = []
    point_y_start = ColumnKeyPoints["whitetop"][1] - 20
    point_x_start = RowKeyPoints["whiteleft"][0] + 5
    point_x_end = RowKeyPoints["whiteright"][0] - 5
    for point_x in range(point_x_start, point_x_end, 5):
        ColumnPointsTmp = getColumnPoint(img_gray, point_x, point_y_start)
        ColumnPointsDictList.append(ColumnPointsTmp)

    # ________ Column 拟合直线
    lineTypeList = ["whitetop", "blacktop", "blackbottom", "whitebottom"]
    for lineType in lineTypeList:
        result = fitLine(ColumnPointsDictList, lineType)
        resultDict[lineType + "PointsList"] = result["PointsList"]
        resultDict[lineType + "Line"] = result["Line"]

    # 更新ColumnKey点
    whitetop_k = resultDict["whitetopLine"][1]/resultDict["whitetopLine"][0]
    whitetop_b = resultDict["whitetopLine"][3] - whitetop_k * resultDict["whitetopLine"][2]
    ColumnKeyPoints["whitetop"][1] = int(whitetop_k * ColumnKeyPoints["whitetop"][1] + whitetop_b)

    blackbottom_k = resultDict["blacktopLine"][1]/resultDict["blacktopLine"][0]
    blackbottom_b = resultDict["blacktopLine"][3] - blackbottom_k * resultDict["blacktopLine"][2]
    ColumnKeyPoints["blacktop"][1] = int(blackbottom_k * ColumnKeyPoints["blacktop"][1] + blackbottom_b)

    # ........ Row 获取点列表
    RowPointsDictList = []
    point_y_start = ColumnKeyPoints["whitetop"][1] + 5
    point_y_end = ColumnKeyPoints["blacktop"][1] - 5
    for point_y in range(point_y_start, point_y_end, 5):
        RowPointsTmp = getRowWhitePoint(img_gray, point_y)
        RowPointsDictList.append(RowPointsTmp)

    # ________ Row 拟合直线
    lineTypeList = ["whiteleft", "whiteright"]
    for lineType in lineTypeList:
        result = fitLine(RowPointsDictList, lineType)
        resultDict[lineType + "PointsList"] = result["PointsList"]
        resultDict[lineType + "Line"] = result["Line"]

    #  ........ Ear 获取点列表
    EarPointsList = []
    point_y_start = ColumnKeyPoints["whitetop"][1]
    point_x_start = RowKeyPoints["whiteright"][0] + 10
    point_x_end = RowKeyPoints["whiteright"][0] + 60
    for Points_x in range(point_x_start, point_x_end, 5):
        EarPointsTmp = getEarPoint(img_gray, Points_x, point_y_start)
        EarPointsList.append(EarPointsTmp)

    # ________ Ear 拟合直线
    lineTypeList = ["top", "bottom"]
    for lineType in lineTypeList:
        result = fitLine(EarPointsList, lineType)
        resultDict[lineType + "PointsList"] = result["PointsList"]
        resultDict[lineType + "Line"] = result["Line"]

    # 计算交点
    intersectionNameList = ["whitetop_whiteleft", "whitetop_whiteright",
                            "whitebottom_whiteleft", "whitebottom_whiteright",
                            "whiteright_top","whiteright_bottom",
                            ]
    for intersectionName in intersectionNameList:
        names = intersectionName.split("_")
        line1 = resultDict.get(names[0] + "Line", [])
        line2 = resultDict.get(names[1] + "Line", [])
        angle_, point_ = line_angle_out(line1, line2)
        point_info = {}
        point_info["angle"] = round(angle_, 1)
        point_info["point"] = point_
        resultDict[intersectionName + "Intersection"] = point_info

    return resultDict

def imgdrawResult(img_gray, resultDict):
    for resultKey in resultDict:
        resultItem = resultDict[resultKey]
        if resultKey[-10:] == "PointsList":
            for point in resultItem:
                cv2.circle(img_gray, point, 5, (0, 0, 255), -1)
        elif resultKey[-4:] == "Line":
            lk = resultItem[1] / resultItem[0]
            lb = resultItem[3] - lk * resultItem[2]
            try:
                cv2.line(img_gray, (0, int(lb)), (int(resultItem[2]), int(resultItem[3])), (255, 255, 0), 1)
            except Exception as e:
                cv2.line(img_gray, (-int(lb/lk), 0), (int(resultItem[2]), int(resultItem[3])), (255, 255, 0), 1)
        elif resultKey[-12:] == "Intersection":
            cv2.circle(img_gray, resultItem["point"], 5, (0, 255, 255), 4, -1)
            cv2.putText(img_gray, str(resultItem["angle"]), resultItem["point"], cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
if __name__ == '__main__':
    start_time = time.time()
    img_path = r"D:\04DataSets\04\box.jpg"
    img_bgr = cv2.imread(img_path)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    ColumnPoints = getColumnPoint(img_gray, 220)     # 122

    RowPointsOne_y = ColumnPoints["whitetop"][1] + int((ColumnPoints["blacktop"][1] - ColumnPoints["whitetop"][1])/2)
    RowPointsOne = getRowWhitePoint(img_gray, RowPointsOne_y)
    RowPointsTwo_y = ColumnPoints["blackbottom"][1] + int((ColumnPoints["whitebottom"][1] - ColumnPoints["blackbottom"][1])/2)
    RowPointsTwo = getRowWhitePoint(img_gray, RowPointsTwo_y)

    # 耳朵
    EarPointsOne_x = RowPointsOne["whiteright"][0] + 10
    EarPointsOne = getEarPoint(img_gray, EarPointsOne_x)
    EarPointsTwo_x = RowPointsOne["whiteright"][0] + 60
    EarPointsTwo = getEarPoint(img_gray, EarPointsTwo_x)
    EarPointsList = []
    for i in range(10, 61, 5):
        EarPointsTwo_x = RowPointsOne["whiteright"][0] + i
        EarPointsTwo = getEarPoint(img_gray, EarPointsTwo_x)
        EarPointsList.append(EarPointsTwo)

    # 上
    earTopPointsList = []
    for EarPoints in EarPointsList:
        earTopPointsList.append(EarPoints["top"])
    earTopPointsList = np.array(earTopPointsList)
    output = cv2.fitLine(earTopPointsList, cv2.DIST_L2, 0, 0.01, 0.01)

    line1 = [RowPointsOne["whiteright"], RowPointsTwo["whiteright"]]
    line2 = [EarPointsOne["top"], EarPointsTwo["top"]]
    angle, point = line_angle(line1, line2)

    line1 = [RowPointsOne["right"], RowPointsTwo["right"]]
    line2b = [EarPointsOne["bottom"], EarPointsTwo["bottom"]]
    angleb, pointb = line_angle(line1, line2b)

    end_time = time.time()
    total_time = end_time - start_time
    print("运行时间为：", total_time*1000, "ms")


    pointDictList = [ColumnPoints, RowPointsOne, RowPointsTwo, EarPointsOne, EarPointsTwo]
    for pointDict in pointDictList:
        for item in pointDict.items():
            cv2.circle(img_bgr, item[1], 5, (0,0,255), -1)
    cv2.circle(img_bgr, point, 5, (0, 255, 255), -1)
    cv2.circle(img_bgr, pointb, 5, (0, 255, 255), -1)

    # 直线
    k = output[1] / output[0]
    b = output[3] - k * output[2]
    cv2.line(img_bgr, (0, int(b)), (600, int(k * 600 + b)), (255, 255, 0), 1)

    top = ColumnPoints["whitetop"][1] - 20
    bottom = ColumnPoints["whitebottom"][1] + 20
    cv2.imshow("point", img_bgr[top:bottom,:])
    print(angle)
    print(angleb)
    cv2.waitKey(0)


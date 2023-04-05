"""
    霍夫直线检测
"""
import math
import cv2 as cv
import numpy as np

def fun_HoughLines():
    img_bgr = cv.imread(cv.samples.findFile(filename))    # 读图
    cv.imshow("img_bgr", img_bgr)
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    img_canny = cv.Canny(img_gray, 50, 200, None, 3)                            # Canny 边缘检测
    # image, rho, theta, threshold, lines=None, srn=None, stn=None, min_theta=None, max_theta=None
    # rho, theta: 分辨率，一般都是这个值
    # threshold : 得票阈值。票数高于这个才认为是直线。由于投票数取决于线上的点数，所以它代表了应该被检测到的线的最小点数。
    # lines :
    # srn 和 stn ：可选参数，用于控制高斯平滑过程的内核大小。
    ###返回  lines [[[rho, rho]]]
    # - 用于存储检测到的直线参数的向量，
    # - 每个向量内存储的是Detla和Theta，即直线的一般式参数，
    # - 可以通过以下公式获得Hough直线上的两个点的坐标：
    #       x = cos(theta) * delta
    #       y = sin(theta) * delta
    lines = cv.HoughLines(img_canny, 1, np.pi / 180, 150, None, 0, 0)           # 霍夫直线检测
    # 显示直线
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(img_bgr, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
    cv.imshow("cv.Canny", img_canny)
    cv.imshow("cv.HoughLines", img_bgr)
    cv.waitKey()
    return 0

def HoughLineTransform():
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # Canny 边缘检测
    dst = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    #
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    dir_root = r"D:\00myGitHub\opencv\samples\data/"
    filename = dir_root + 'sudoku.png'
    fun_HoughLines()
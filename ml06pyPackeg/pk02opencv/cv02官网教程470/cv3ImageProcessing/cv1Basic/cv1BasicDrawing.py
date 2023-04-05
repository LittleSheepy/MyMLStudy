"""
    画图
"""
import math
import cv2
import cv2 as cv
import numpy as np

"""
LineTypes 可选的值包括：
- cv2.LINE_AA
- cv2.LINE_4
- cv2.LINE_8
- cv2.FILLED
"""
# 画椭圆
def my_ellipse():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    # def ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=None, lineType=None, shift=None):
    # axes : 椭圆主轴大小的一半。
    # thickness : 线粗细 -1 填充。
    cv.ellipse(image, (200, 200), (100, 25), 45, 0, 180, (255, 0, 0), 5, cv2.LINE_8)
    cv.imshow("ellipse", image)
    cv.waitKey(0)

# 画圆
def my_circle():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(image, (200, 200), 100, (255, 0, 0), 5, cv2.LINE_8)
    cv.imshow("circle", image)
    cv.waitKey(0)

# 填充轮廓
def my_fillPoly():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    points = np.array([[100, 100], [200, 100],
                                [200, 200], [300, 200],
                                            [300, 300],
                    [100, 300]], np.int32)
    # def fillPoly(img, pts, color, lineType=None, shift=None, offset=None):
    cv.fillPoly(image, [points], (255, 255, 255), cv2.LINE_8)
    cv.imshow("fillPoly", image)
    cv.waitKey(0)

# 画线
def my_line():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.line(image, (100, 200), (300, 200), (255, 0, 0), 1, cv2.LINE_8)
    cv.imshow("circle", image)
    cv.waitKey(0)

# 画框
def my_rectangle():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (300, 300), (255, 0, 0), 1, cv2.LINE_8)
    cv.imshow("circle", image)
    cv.waitKey(0)

if __name__ == "__main__":
    # dir_root = r"D:\00myGitHub\opencv\samples\data/"
    # filename = dir_root + 'sudoku.png'
    W = 400
    my_rectangle()
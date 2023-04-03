import cv2
import numpy as np
# 读取图片并转换为灰度图像
img = cv2.imread(r'D:\04DataSets\ningjingLG\black\black_0219193_CM1_4.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 应用高斯模糊以减少噪音
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# 检测边缘
edges = cv2.Canny(blur, 30, 90, apertureSize=3)
cv2.imwrite('edges.jpg', edges)
# 应用霍夫直线变换以检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
# 绘制检测到的直线
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x1 - x2)*10 < abs(y1 - y2):
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# 显示结果
cv2.imwrite('image100.jpg', img)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
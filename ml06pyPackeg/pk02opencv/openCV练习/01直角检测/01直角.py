
import cv2
# 读取图片并灰度化
image_path = r'D:\04DataSets\04\box.jpg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret0, gray_thre = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('gray_thre', gray_thre)
# 边缘检测
edges = cv2.Canny(gray_thre, 50, 150, apertureSize=3)
cv2.imshow('edges', edges)
# 轮廓检测
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 遍历轮廓，筛选出近似矩形
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        # 计算近似矩形的边长并判断是否为直角
        x, y, w, h = cv2.boundingRect(approx)
        if abs(w - h) < min(w, h) * 0.2:
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
# 展示结果图片
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
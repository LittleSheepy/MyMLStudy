import cv2
import numpy as np
# 读取原始图像
dir_root = r"D:\02dataset\02opencv_data/"
img_path = dir_root + 'HappyFish.jpg'
img = cv2.imread(img_path)
# 将图像转为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 对灰度图像进行二值化处理
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
# 填充内部空洞
mask = cv2.floodFill(thresh, None, (0, 0), 255)
# 获取轮廓
contours, hierarchy = cv2.findContours(np.int32(mask[1]), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_NONE)
# 循环遍历轮廓并绘制
for i, contour in enumerate(contours):
    cv2.drawContours(img, contours, i, (0, 0, 255), 2)
# 显示结果
cv2.imshow("image", img)
cv2.waitKey(0)
import cv2
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread('defectBig.png')
img = cv2.imread('template2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.imshow(thresh, cmap='gray')
plt.show()
print(thresh.shape)

# 噪声去除
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# 确定背景区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 膨胀
# 寻找前景区域-对象分离
# separate分离系数，取值范围0.1-1
separate = 0.4
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, separate * dist_transform.max(), 255, 0)  # sure_fg为分离对象的图像
# 找到未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 类别标记
ret, markers = cv2.connectedComponents(sure_fg)
# 为所有的标记加1，保证背景是0而不是1
markers = markers+1
# 现在让所有的未知区域为0
markers[unknown==255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

dist_transform = cv2.normalize(dist_transform, 0, 1.0, cv2.NORM_MINMAX) * 80
cv2.imshow("dist_transform", dist_transform)

cv2.imshow("img", img)
cv2.waitKey()
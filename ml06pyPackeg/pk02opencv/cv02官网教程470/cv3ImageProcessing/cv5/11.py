import cv2
import numpy as np

# 读取图像
src = cv2.imread('input.jpg')

# 将输入图像转换为float32类型
sharp = np.float32(src)

# 对图像进行拉普拉斯变换
imgLaplacian = cv2.Laplacian(sharp, cv2.CV_32F, ksize=3)

# 将imgLaplacian图像中的像素值限制在0到255之间
imgLaplacian = np.clip(imgLaplacian, 0, 255)

# 将sharp图像和imgLaplacian图像相减，得到一个新的图像imgResult
imgResult = sharp - imgLaplacian

# 对图像进行二值化处理
imgResult = np.uint8(imgResult)
ret, thresh = cv2.threshold(imgResult, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 对图像进行形态学操作
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 进行距离变换
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# 执行分水岭算法
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(opening, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown==255] = 0
markers = cv2.watershed(src, markers)
src[markers == -1] = [255,0,0]

# 显示结果
cv2.imshow('Result', src)
cv2.waitKey(0)
cv2.destroyAllWindows()
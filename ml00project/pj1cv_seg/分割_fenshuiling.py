"""
Author:XiaoMa
date:2021/11/2
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path = r"D:\04DataSets\ningjingLG\all\/black_0074690_CM1_2.bmp"
img0 = cv2.imread(img_path)
img1 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
h, w = img1.shape[:2]
print(h, w)
cv2.namedWindow("W0")
cv2.imshow("W0", img1)
cv2.waitKey(delay=0)
# 图像进行二值化
##第一种阈值类型
ret0, img3 = cv2.threshold(img2, 15, 255, cv2.THRESH_BINARY)
print(ret0)
##第二种阈值类型
ret1, img4 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY_INV)
print(ret1)
##第三种阈值类型
ret2, img5 = cv2.threshold(img2, 130, 255, cv2.THRESH_TRUNC)
print(ret2)
##第四种阈值类型
ret3, img6 = cv2.threshold(img2, 128, 255, cv2.THRESH_TOZERO)
print(ret3)
##第五种阈值类型
ret4, img7 = cv2.threshold(img2, 130, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
print(ret4)
# 将所有阈值类型得到的图像绘制到同一张图中
plt.rcParams['font.family'] = 'SimHei'  # 将全局中文字体改为黑体
figure = [img2, img3, img4, img5, img6, img7]
title = ["原图", "第一种阈值类型", "第二种阈值类型", "第三种阈值类型", "第四种阈值类型", "第五种阈值类型"]
for i in range(6):
    figure[i] = cv2.cvtColor(figure[i], cv2.COLOR_BGR2RGB)  # 转化图像通道顺序，这一个步骤要记得
    plt.subplot(3, 2, i + 1)
    plt.imshow(figure[i])
    plt.title(title[i])  # 添加标题
plt.savefig("\cvtwelven.jpg")  # 保存图像，如果不想保存也可删去这一行
plt.show()
# 边缘检测之Sobel 算子
img8 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)
cv2.namedWindow("W1")
cv2.imshow("W1", img8)
cv2.waitKey(delay=0)
# K-means均值聚类
Z = img1.reshape((-1, 3))
Z = np.float32(Z)  # 转化数据类型
c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 4
ret, label, center = cv2.kmeans(Z, k, None, c, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
img9 = res.reshape((img1.shape))
cv2.namedWindow("W2")
cv2.imshow("W2", img9)
cv2.waitKey(delay=0)

# 分水岭算法
ret1, img10 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # （图像阈值分割，将背景设为黑色）
cv2.namedWindow("W3")
cv2.imshow("W3", img10)
cv2.waitKey(delay=0)
##noise removal（去除噪声，使用图像形态学的开操作，先腐蚀后膨胀）
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(img10, cv2.MORPH_OPEN, kernel, iterations=2)
# sure background area(确定背景图像，使用膨胀操作)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Finding sure foreground area（确定前景图像，也就是目标）
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret2, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
# Finding unknown region（找到未知的区域）
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# Marker labelling
ret3, markers = cv2.connectedComponents(sure_fg)  # 用0标记所有背景像素点
# Add one to all labels so that sure background is not 0, but 1（将背景设为1）
markers = markers + 1
##Now, mark the region of unknown with zero（将未知区域设为0）
markers[unknown == 255] = 0
markers = cv2.watershed(img1, markers)  # 进行分水岭操作
img1[markers == -1] = [0, 0, 255]  # 边界区域设为-1，颜色设置为红色
cv2.namedWindow("W4")
cv2.imshow("W4", img1)
cv2.waitKey(delay=0)

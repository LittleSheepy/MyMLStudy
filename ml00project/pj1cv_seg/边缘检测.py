"""
Author:XiaoMa
date:2021/10/29
"""
import cv2
import matplotlib.pyplot as plt
#读取图像信息
img0 = cv2.imread("./defect2.png")
img1 = cv2.resize(img0, dsize = None, fx = 0.5, fy = 0.5)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("E:\From Zhihu\For the desk\cvtenGray.jpg", img2)   #保存灰度图
h, w = img1.shape[:2]
print(h, w)
cv2.namedWindow("W0")
cv2.imshow("W0", img1)
cv2.waitKey(delay = 0)
#canny 算子
img4 = cv2.Canny(img2, 100, 200)
cv2.namedWindow("W4")
cv2.imshow("W4", img4)
cv2.waitKey(delay = 0)
#Sobel 算子
img3 = cv2.Sobel (img2, cv2.CV_64F, 0, 1, ksize=5)
cv2.namedWindow("W3")
cv2.imshow("W3", img3)
cv2.waitKey(delay = 0)
#Laplacian 算子
img7 = cv2.Laplacian(img2, cv2.CV_64F)
cv2.namedWindow("W7")
cv2.imshow("W7", img7)
cv2.waitKey(delay = 0)
#Scharr 算子
img9 = cv2.Scharr(img2, cv2.CV_64F, 0, 1)
cv2.namedWindow("W9")
cv2.imshow("W9", img9)
cv2.waitKey(delay = 0)
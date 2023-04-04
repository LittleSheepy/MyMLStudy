import numpy as np
import cv2
# 读取图像
image_path = r'D:\04DataSets\04\box_center.jpg'
img = cv2.imread(image_path, 0)
# 计算x和y方向的梯度
grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# 计算梯度的幅值和方向
grad_mag, grad_angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
# 显示图像和梯度
cv2.imshow('Original Image', img)
cv2.imshow('Gradient X', grad_x)
cv2.imshow('Gradient Y', grad_y)
cv2.imshow('Gradient Magnitude', grad_mag)
cv2.imshow('Gradient Angle', grad_angle)
cv2.waitKey(0)
cv2.destroyAllWindows()
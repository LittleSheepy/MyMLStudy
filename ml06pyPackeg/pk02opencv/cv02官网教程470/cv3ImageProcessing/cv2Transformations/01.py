# -*- coding: utf-8 -*-

import cv2

import numpy as np


def hough_detectline(img):
    thetas = np.deg2rad(np.arange(0, 180))
    row, cols = img.shape
    diag_len = np.ceil(np.sqrt(row ** 2 + cols ** 2))
    rhos = np.linspace(-diag_len, diag_len, int(2 * diag_len))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_theta = len(thetas)
    # vote
    vote = np.zeros((int(2 * diag_len), num_theta), dtype=np.uint64)
    y_inx, x_inx = np.nonzero(img)
    # vote in hough space
    for i in range(len(x_inx)):
        x = x_inx[i]
        y = y_inx[i]

        for j in range(num_theta):
            rho = round(x * cos_t[j] + y * sin_t[j]) + diag_len
            if isinstance(rho, int):
                vote[rho, j] += 1
            else:
                vote[int(rho), j] += 1
    return vote, rhos, thetas

# image = cv2.imread(r'C:\Users\Y\Desktop\input_0.png')

# image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# image_binary=cv2.Canny(image_gray,150,255)

image = np.zeros((500, 500))
image_bgr = np.zeros((500, 500, 3))
image1 = np.zeros((500, 500, 3))

image[10:100, 10:100] = np.eye(90)
image_bgr[10:100, 10:100,2] = image_bgr[10:100, 10:100,2] + np.eye(90)*255
image_bgr[8:98, 10:100,2] = image_bgr[8:98, 10:100,2] + np.eye(90)*255
image_bgr[9:99, 10:100,2] = image_bgr[9:99, 10:100,2] + np.eye(90)*255
image_bgr[11:101, 10:100,2] = image_bgr[11:101, 10:100,2] + np.eye(90)*255
image_bgr[12:102, 10:100,2] = image_bgr[12:102, 10:100,2] + np.eye(90)*255

cv2.imwrite("image_bgr.jpg", image_bgr)
accumulator, rhos, thetas = hough_detectline(image)

# look for peaks
s = np.argsort(accumulator, axis=[0,1])
idx = np.argmax(accumulator)

rho = rhos[int(idx / accumulator.shape[1])]

theta = thetas[idx % accumulator.shape[1]]

k = -np.cos(theta) / np.sin(theta)

b = rho / np.sin(theta)

x = np.float32(np.arange(10, 80, 2))

# 要在image 上画必须用float32，要不然会报错(float不行)

y = np.float32(k * x + b)

cv2.imshow("original image", image)

for i in range(len(x) - 1):
    try:
        cv2.circle(image_bgr, (int(x[i]), int(y[i])), 1, (255, 0, 0), 1)
        pass
    except Exception as e:
        print(e)

cv2.imwrite("hough.png", image_bgr)
cv2.imshow("hough", image_bgr)

print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))
cv2.waitKey(0)
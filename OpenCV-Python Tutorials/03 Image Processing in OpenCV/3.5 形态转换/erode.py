import cv2 as cv
import numpy as np

img = cv.imread('j.png', cv.IMREAD_GRAYSCALE)
kernel = np.ones((5,5), np.uint8)
erosion = cv.erode(img, kernel, iterations=1)
cv.imshow('erode',np.hstack((img, erosion)))
cv.waitKey(0)
import pysift
import numpy as np
import cv2 as cv
imgName = 'home.jpg'
#imgName = 'littlelight.png'
img = cv.imread(imgName)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

kp, des = pysift.computeKeypointsAndDescriptors(gray)

img = cv.drawKeypoints(gray, kp, img)

cv.imwrite('sift'+imgName, img)
cv.imshow('sift'+imgName, img)
cv.waitKey(0)


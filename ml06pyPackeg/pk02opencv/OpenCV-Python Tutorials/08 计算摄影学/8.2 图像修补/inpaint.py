import numpy as np
import cv2 as cv

img = cv.imread('messi_2.jpg')
mask = cv.imread('mask2.png',cv.IMREAD_GRAYSCALE)

dst = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)

cv.imshow('dst',np.hstack((img, dst)))
cv.waitKey(0)
cv.destroyAllWindows()
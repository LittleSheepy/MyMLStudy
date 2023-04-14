import cv2 as cv
import numpy as np

def nothing(x):
    pass
cv.namedWindow('line',cv.WINDOW_NORMAL)
cv.createTrackbar('thickness','line',1,20, nothing)
while True:
    thickness = cv.getTrackbarPos('thickness','line')
    if thickness == -1:break
    if thickness == 0:
        cv.setTrackbarPos('thickness','line',1)
        continue
    # Create a black image
    img = np.zeros((512, 512, 3), np.uint8)
    # Draw a diagonal blue line with thickness of 5 px
    cv.line(img, (0, 0), (511, 511), (255, 0, 0), thickness=thickness)
    cv.imshow('line',img)
    cv.waitKey(1)




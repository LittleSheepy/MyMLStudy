import numpy as np
import cv2 as cv

def nothing(x):
    pass

cv.namedWindow('goodFeaturesToTrack', cv.WINDOW_NORMAL)
cv.createTrackbar('maxCorners', 'goodFeaturesToTrack', 0, 50, nothing)
cv.createTrackbar('qualityLevel', 'goodFeaturesToTrack', 10, 999, nothing)
cv.createTrackbar('minDistance', 'goodFeaturesToTrack', 10, 50, nothing)
while True:
    maxCorners = cv.getTrackbarPos('maxCorners','goodFeaturesToTrack')
    qualityLevel = cv.getTrackbarPos('qualityLevel','goodFeaturesToTrack')
    minDistance = cv.getTrackbarPos('minDistance','goodFeaturesToTrack')

    img = cv.imread('chessboard-3.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, maxCorners, 0.001 * qualityLevel, minDistance)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv.circle(img, (x,y), 3, 255, -1)
    cv.imshow('goodFeaturesToTrack',img)
    cv.waitKey(1)

cv.destroyAllWindows()

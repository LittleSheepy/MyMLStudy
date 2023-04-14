import cv2 as cv
import numpy as np

flgsEnum = {
    0:'INPAINT_NS',
    1:'INPAINT_TELEA'
}
def nothing(x):
    pass
cv.namedWindow('inpaint',cv.WINDOW_NORMAL)
cv.createTrackbar('inpaintRadius','inpaint',0,10, nothing)
cv.createTrackbar('flags','inpaint',0,1, nothing)
img = cv.imread('messi_2.jpg')
mask = cv.imread('mask2.png', 0)

while True:
    inpaintRadius = cv.getTrackbarPos('inpaintRadius','inpaint')
    flags = cv.getTrackbarPos('flags','inpaint')
    if inpaintRadius == -1 or flags == -1:break
    des = cv.inpaint(img, mask, inpaintRadius, flags)
    cv.putText(des, flgsEnum[flags], org=(10, 30), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5,
               color=(255, 255, 255), lineType=cv.LINE_4)
    cv.imshow('inpaint', np.hstack((img, des)))
    cv.waitKey(1)

cv.destroyAllWindows()
import cv2 as cv
import cv2
import numpy as np
WindowName = "HSV_Test"

def nothing(x):
    pass
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    #r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    r = np.array([hgain, sgain, vgain]) + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
cv.namedWindow(WindowName,cv.WINDOW_NORMAL)
cv.createTrackbar('H',WindowName,0,360, nothing)
cv.createTrackbar('S',WindowName,0,2000, nothing)
cv.createTrackbar('V',WindowName,0,2000, nothing)
while True:
    H = cv.getTrackbarPos('H',WindowName)
    S = cv.getTrackbarPos('S',WindowName)
    V = cv.getTrackbarPos('V',WindowName)
    if H == -1:break
    img = cv.imread('tongli.jpg', cv.IMREAD_COLOR)
    img_org = img.copy()
    h = H%360 - 180
    s = S/1000 - 1
    v = V/1000 - 1
    augment_hsv(img, h, s, v)

    img_heng = np.concatenate((img_org, img), axis=1)
    cv.imshow(WindowName,img_heng)
    cv.waitKey(1)

cv.destroyAllWindows()




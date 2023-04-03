import cv2
import numpy as np
img = cv2.imread(r'D:\04DataSets\ningjingLG\black\black_0074690_CM1_1.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow('edges', edges)
lines = cv2.HoughLines(edges, 1, np.pi/2, 150)
#lines = cv2.HoughLinesP(edges, 1, np.pi/2, 150, minLineLength=10, maxLineGap=10)
for line in lines:
    rho, theta = line[0]
    if abs(abs(theta) - (np.pi / 2.0)) < 0.1: #  or abs(abs(theta) -(3.0 * np.pi / 2.0))  < 0.1
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = b*rho
        y0 = a*rho
        x1 = int(x0 + 1000*(a))
        y1 = int(y0 + 1000*(b))
        x2 = int(x0 - 1000*(a))
        y2 = int(y0 - 1000*(b))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
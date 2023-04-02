"""
    击中-击不中（匹配）  1:255 -1:0  0:忽略
"""
import cv2 as cv
import numpy as np

input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 0, 0]), dtype="uint8")
input_image = input_image * 255

kernel = np.array((
        [0, 1, 0],
        [1, -1, 1],
        [0, 1, 0]), dtype="int")
# 1
kernel1 = np.array((
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]), dtype="int")
# 2
kernel2 = np.array((
        [0, 1, 0],
        [1, -1, 1],
        [1, 1, 1]), dtype="int")

kernel = kernel_dilating
output_image = cv.morphologyEx(input_image, cv.MORPH_HITMISS, kernel)

rate = 50
kernel = (kernel + 1) * 127
kernel = np.uint8(kernel)

kernel = cv.resize(kernel, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
cv.imshow("kernel", kernel)
cv.moveWindow("kernel", 0, 0)

input_image = cv.resize(input_image, None, fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
cv.imshow("Original", input_image)
cv.moveWindow("Original", 0, 200)

output_image = cv.resize(output_image, None , fx = rate, fy = rate, interpolation = cv.INTER_NEAREST)
cv.imshow("Hit or Miss", output_image)
cv.moveWindow("Hit or Miss", 500, 200)

cv.waitKey(0)
cv.destroyAllWindows()

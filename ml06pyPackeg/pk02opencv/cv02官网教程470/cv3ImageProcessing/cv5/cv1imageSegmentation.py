"""
    基于距离变换和分水岭算法的图像分割
"""
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

dir_root = r"D:\02dataset\02opencv_data/"
img_path = dir_root + "cards.png"
img_bgr = cv.imread(cv.samples.findFile(img_path))

img_bgr_no_bg = img_bgr.copy()
img_bgr_no_bg[np.all(img_bgr == 255, axis=2)] = 0               # 背景置0

cv.imshow('Source Image', img_bgr)
cv.imshow('Black Background Image', img_bgr_no_bg)

## [sharp]
# Create a kernel that we will use to sharpen our image
# an approximation of second derivative, a quite strong kernel
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
imgLaplacian = cv.filter2D(img_bgr_no_bg, cv.CV_32F, kernel)
sharp = np.float32(img_bgr_no_bg)
imgResult = sharp - imgLaplacian

cv.imshow('imgLaplacian', imgLaplacian)
cv.imshow('sharp', sharp)
cv.imshow('imgResult', imgResult)
# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
cv.imshow('imgResult_clip', imgResult)
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)
cv.imshow('imgLaplacian_clip', imgLaplacian)


#cv.imshow('Laplace Filtered Image', imgLaplacian)
#cv.imshow('New Sharped Image', imgResult)
## [sharp]

## [bin]
# Create binary image from source image
bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
_, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow('Binary Image', bw)
## [bin]
## [dist]
# Perform the distance transform algorithm
dist = cv.distanceTransform(bw, cv.DIST_L2, 3)

# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
cv.imshow('Distance Transform Image', dist)
## [dist]

## [peaks]
# Threshold to obtain the peaks
# This will be the markers for the foreground objects
_, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)

cv.imshow('dist', dist)
# Dilate a bit the dist image
kernel1 = np.ones((3,3), dtype=np.uint8)
dist = cv.dilate(dist, kernel1)
cv.imshow('Peaks', dist)
## [peaks]

## [seeds]
# Create the CV_8U version of the distance image
# It is needed for findContours()
dist_8u = dist.astype('uint8')

# Find total markers
contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create the marker image for the watershed algorithm
markers = np.zeros(dist.shape, dtype=np.int32)

# Draw the foreground markers
for i in range(len(contours)):
    cv.drawContours(markers, contours, i, (i+1), -1)

# Draw the background marker
# cv.circle(markers, (5,5), 3, (255,255,255), -1)
cv.circle(markers, (5,5), 3, 255, -1)
cv.imshow('Markers',  markers*255/len(contours))
## [seeds]

## [watershed]
# Perform the watershed algorithm
cv.watershed(imgResult, markers)
cv.imshow('watershed imgResult', imgResult)
cv.imshow('watershed markers', markers*255/len(contours))

#mark = np.zeros(markers.shape, dtype=np.uint8)
mark = markers.astype('uint8')
mark = cv.bitwise_not(mark)
# uncomment this if you want to see how the mark
# image looks like at that point
#cv.imshow('Markers_v2', mark)

# Generate random colors
colors = []
for contour in contours:
    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))

# Create the result image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

# Fill labeled objects with random colors
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i,j]
        if index > 0 and index <= len(contours):
            dst[i,j,:] = colors[index-1]

# Visualize the final image
cv.imshow('Final Result', dst)
## [watershed]

cv.waitKey()

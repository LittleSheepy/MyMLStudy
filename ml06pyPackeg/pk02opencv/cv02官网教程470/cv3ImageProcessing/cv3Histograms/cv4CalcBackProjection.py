import cv2 as cv
import numpy as np
# 背投影
def my_calcBackProject(img_hsv, histSize, ranges):
    hist = cv.calcHist([img_hsv], [0], None, [histSize], ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    backproj = cv.calcBackProject([img_hsv], [0], hist, ranges, scale=1)
    return hist, backproj

def Hist_and_Backproj(val):
    bins = val
    histSize = max(bins, 2)
    ranges = [0, 180]

    hist, backproj = my_calcBackProject(hsv, histSize, ranges)

    h = 400
    bin_w = int(round(400 / histSize))
    histImg = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(bins):
        cv.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(np.round( hist[i]*h/255.0 )) ), (0, 0, 255), cv.FILLED)
    cv.imshow('BackProj', backproj)
    cv.imshow('Histogram', histImg)
if __name__ == '__main__':
    dir_root = r"D:\02dataset\02opencv_data/"
    filename = dir_root + 'hand0.jpg'
    src = cv.imread(filename)
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

    window_image = 'Source image'
    cv.namedWindow(window_image)
    bins = 13
    cv.createTrackbar('bins: ', window_image, bins, 180, Hist_and_Backproj )
    Hist_and_Backproj(bins)

    cv.imshow(window_image, src)
    cv.waitKey()

import cv2 as cv
import numpy as np

# 漫水填充(涂油漆桶)
def my_floodFill(img_bgr=None, seed=(100, 100),newVal=(120,)*3,low=30, up=30):
    img_bgr = cv.imread(filename) if img_bgr is None else img_bgr
    mask2 = np.zeros((img_bgr.shape[0] + 2, img_bgr.shape[1] + 2), dtype=np.uint8)
    newMaskVal = 255
    connectivity = 8
    flags = connectivity + (newMaskVal << 8 ) + cv.FLOODFILL_FIXED_RANGE + cv.FLOODFILL_MASK_ONLY
    # def floodFill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None, flags=None):
    # seedPoint : 开始填充的点
    # newVal : 想要填充的新值
    # loDiff : 填充时，newVal所处区域内的最低色差值
    # upDiff : 填充时，newVal所处区域内的最高色差值
    # flags  : 表示填充方式。
    #             cv2.FLOODFILL_FIXED_RANGE：使用固定的范围而不是浮动范围。  b0001 0000 0000 0000 0000
    #             cv2.FLOODFILL_MASK_ONLY：只填充掩码而不是图像。           b0010 0000 0000 0000 0000
    #             (newMaskVal << 8 )                                          b1111 1111 0000 0000
    #             8                                                                          b1000
    """ 漫水填充 """
    cv.floodFill(img_bgr, mask2, seed, newVal, (low,)*3, (up,)*3, flags)
    mask = mask2[1:-1,1:-1]
    return mask

# 背投影
def my_calcBackProject(images, mask=None, channels=[0], histSize=[256], ranges=[0, 256]):
    hist = cv.calcHist(images, channels, mask, histSize, ranges, accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    backproj = cv.calcBackProject(images, channels, hist, ranges, scale=1)
    return hist, backproj

low, up = 20, 20
def callback_low(val):
    global low
    low = val

def callback_up(val):
    global up
    up = val

def pickPoint(event, x, y, flags, param):
    if event != cv.EVENT_LBUTTONDOWN: return
    mask = my_floodFill(img_bgr, seed=(x, y), newVal=(120,)*3, low=low, up=up)

    hist, backproj = my_calcBackProject([hsv], mask, channels=[0, 1], histSize=[30, 32], ranges=[0, 180, 0, 256])
    cv.imshow('Mask&BackProj', cv.hconcat([mask, backproj]))

def Hist_and_Backproj(bins):
    histSize = max(bins, 2)
    hist, backproj = my_calcBackProject([hsv], channels=[0], histSize=[histSize], ranges=[0, 180])

    w, h = 400, 400
    bin_w = int(round(w / histSize))
    histImg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(bins):
        cv.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(np.round( hist[i]*h/255.0 )) ), (0, 0, 255), cv.FILLED)
    cv.imshow('BackProj', backproj)
    cv.imshow('Histogram', histImg)

def TrackbarTest():
    window_image = 'Source image'
    cv.namedWindow(window_image)
    cv.imshow(window_image, img_bgr)
    cv.createTrackbar('Hist&Backproj', window_image, 13, 180, Hist_and_Backproj)
    cv.createTrackbar('Low thresh', window_image, low, 255, callback_low)
    cv.createTrackbar('High thresh', window_image, up, 255, callback_up)
    cv.setMouseCallback(window_image, pickPoint)


if __name__ == '__main__':
    dir_root = r"D:\02dataset\02opencv_data/"
    filename = dir_root + 'hand0.jpg'
    img_bgr = cv.imread(filename)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    # 漫水填充(涂油漆桶)
    mask = my_floodFill()
    cv.imshow('mask', mask)

    # 背投影
    hist, backproj = my_calcBackProject([hsv], mask=None, channels=[0], histSize=[256], ranges=[0, 256])
    cv.imshow('BackProj', backproj)

    # 滑动条调数测试
    TrackbarTest()

    cv.waitKey()


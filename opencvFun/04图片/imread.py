import cv2 as cv

IMREAD_ENUM={
    'IMREAD_GRAYSCALE'  :0, # 将图像转换为单通道灰度图像
    'IMREAD_COLOR'      :1, # 将图像转换为3通道BGR彩色图像
    'IMREAD_ANYDEPTH'   :2, # 当图像具有相应深度时返回16/32位图像，否则将其转换为8位
    'IMREAD_ANYCOLOR'   :4, # 以任何可能的颜色格式读取图像
}
descrip = {
    0:'IMREAD_GRAYSCALE',
    1:'IMREAD_COLOR',
    2:'IMREAD_ANYDEPTH',
    4:'IMREAD_ANYCOLOR'
}

def nothing(x):
    pass
imreadFlag = cv.IMREAD_COLOR
cv.namedWindow('imread',cv.WINDOW_NORMAL)
cv.createTrackbar('Flag','imread',0,4, nothing)
while True:
    imreadFlag = cv.getTrackbarPos('Flag','imread')
    if imreadFlag == -1:break
    if imreadFlag not in descrip:
        cv.waitKey(1)
        continue
    img = cv.imread('messi5.jpg', imreadFlag)
    cv.putText(img,descrip[imreadFlag] or "",org=(10, 40), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 255, 255),lineType=cv.LINE_4)
    cv.imshow('imread',img)
    cv.waitKey(1)

cv.destroyAllWindows()

import cv2

img = cv2.imread(r"D:\04DataSets\02\images\CB_00001.png")

# 去噪
image = cv2.GaussianBlur(img, (3, 3), 0)

# 转为灰度图
gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ostu阈值分割
ret, th1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

'''轮廓检测与绘制'''
# 检测轮廓(外轮廓)
th1 = cv2.dilate(th1, None)  # 膨胀，保证同一个字符只有一个外轮廓
contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 轮廓可视化
th1_bgr = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)  # 转为三通道图

cv2.drawContours(th1_bgr, contours, -1, (0, 0, 255), 2)  # 轮廓可视化

cv2.imshow("th1_bgr", th1_bgr)

cv2.waitKey()

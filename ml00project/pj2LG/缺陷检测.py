import cv2
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库
import scipy.signal as signal
import numpy as np

img_dir = r"D:\04DataSets\ningjingLG\02ZangWu\LateralPollution\img/"
img_path = img_dir + r'black_0064692_CM1_4.bmp'
# 读取图像
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 模糊处理
img_blur = cv2.blur(img_gray, (5, 5))
img_blur = cv2.blur(img_blur, (5, 5))

# 获取直方图并显示
hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
hist = cv2.calcHist([img_blur], [0], None, [256], [0, 256])

peaks, properties = signal.find_peaks(hist[:, 0], distance=30, height=400, width=2)
valleys, _ = signal.find_peaks(-hist[:, 0], distance=30, height=-400)
# 打印峰值数量
print('直方图中有', len(peaks), '个峰值')
plt.plot(hist_gray[1:])
plt.plot(hist[1:])
plt.show()  # 显示折线图直方图
cv2.imshow("img", cv2.hconcat([img_gray, img_blur]))
cv2.waitKey(0)
#ret, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret, thresh = cv2.threshold(img_blur, 210, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresh, cmap='gray')
plt.show()
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(img_gray,markers)
img_gray[markers == -1] = [255,0,0]

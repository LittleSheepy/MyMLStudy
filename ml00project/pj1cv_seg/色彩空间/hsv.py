import cv2

import numpy as np

from matplotlib import pyplot as plt



#pic_file = '../data/images/image_crocus_0003.png'

pic_file = r"../template2.png"

img_bgr = cv2.imread(pic_file, cv2.IMREAD_COLOR) #OpenCV读取颜色顺序：BGR
img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

img_h = img_hsv[..., 0]

img_s = img_hsv[..., 1]

img_v = img_hsv[..., 2]

fig = plt.gcf() # 分通道显示图片

fig.set_size_inches(10, 15)

plt.subplot(221)

plt.imshow(img_hsv)

plt.axis('off')

plt.title('HSV')

plt.subplot(222)

plt.imshow(img_h, cmap='gray')

plt.axis('off')

plt.title('H')

plt.subplot(223)

plt.imshow(img_s, cmap='gray')

plt.axis('off')

plt.title('S')

plt.subplot(224)

plt.imshow(img_v, cmap='gray')

plt.axis('off')

plt.title('V')

plt.show()

# 按R、G、B三个通道分别计算颜色直方图

h_hist = cv2.calcHist([img_hsv], [0], None, [256], [0, 256])

s_hist = cv2.calcHist([img_hsv], [1], None, [256], [0, 256])

v_hist = cv2.calcHist([img_hsv], [2], None, [256], [0, 256])

# 显示3个通道的颜色直方图

plt.plot(h_hist, label='H', color='blue')

plt.plot(s_hist, label='S', color='green')

plt.plot(v_hist, label='V', color='red')

plt.legend(loc='best')

plt.xlim([0, 256])

plt.show()
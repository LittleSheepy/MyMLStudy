import cv2

import numpy as np

from matplotlib import pyplot as plt



#pic_file = '../data/images/image_crocus_0003.png'

pic_file = r"../template2.png"

img_bgr = cv2.imread(pic_file, cv2.IMREAD_COLOR) #OpenCV读取颜色顺序：BGR

img_b = img_bgr[..., 0]

img_g = img_bgr[..., 1]

img_r = img_bgr[..., 2]

fig = plt.gcf() #图片详细信息

fig = plt.gcf() #分通道显示图片

fig.set_size_inches(10, 15)

plt.subplot(221)

plt.imshow(np.flip(img_bgr, axis=2)) #展平图像数组并显示

plt.axis('off')

plt.title('Image')

plt.subplot(222)

plt.imshow(img_r, cmap='gray')

plt.axis('off')

plt.title('R')

plt.subplot(223)

plt.imshow(img_g, cmap='gray')

plt.axis('off')

plt.title('G')

plt.subplot(224)

plt.imshow(img_b, cmap='gray')

plt.axis('off')

plt.title('B')

plt.show()
# 按R、G、B三个通道分别计算颜色直方图

b_hist = cv2.calcHist([img_bgr], [0], None, [256], [0, 256])

g_hist = cv2.calcHist([img_bgr], [1], None, [256], [0, 256])

r_hist = cv2.calcHist([img_bgr], [2], None, [256], [0, 256])

# 显示3个通道的颜色直方图

plt.plot(b_hist, label='B', color='blue')

plt.plot(g_hist, label='G', color='green')

plt.plot(r_hist, label='R', color='red')

plt.legend(loc='best')

plt.xlim([0, 256])

plt.show()
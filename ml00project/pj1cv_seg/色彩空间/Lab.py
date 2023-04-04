import cv2

import numpy as np

from matplotlib import pyplot as plt

# pic_file = '../data/images/image_crocus_0003.png'

pic_file = r"../template2.png"

img_bgr = cv2.imread(pic_file, cv2.IMREAD_COLOR)  # OpenCV读取颜色顺序：BGR
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

img_ls = img_lab[..., 0]

img_as = img_lab[..., 1]

img_bs = img_lab[..., 2]

# 分通道显示图片

fig = plt.gcf()

fig.set_size_inches(10, 15)

plt.subplot(221)

plt.imshow(img_lab)

plt.axis('off')

plt.title('L*a*b*')

plt.subplot(222)

plt.imshow(img_ls, cmap='gray')

plt.axis('off')

plt.title('L*')

plt.subplot(223)

plt.imshow(img_as, cmap='gray')

plt.axis('off')

plt.title('a*')

plt.subplot(224)

plt.imshow(img_bs, cmap='gray')

plt.axis('off')

plt.title('b*')

plt.show()
# 按R、G、B三个通道分别计算颜色直方图

l_hist = cv2.calcHist([img_lab], [0], None, [256], [0, 256])

a_hist = cv2.calcHist([img_lab], [1], None, [256], [0, 256])

b_hist = cv2.calcHist([img_lab], [2], None, [256], [0, 256])

# 显示3个通道的颜色直方图

plt.plot(l_hist, label='L', color='blue')

plt.plot(a_hist, label='A', color='green')

plt.plot(b_hist, label='B', color='red')

plt.legend(loc='best')

plt.xlim([0, 256])

plt.show()
import numpy as np
import cv2

# 读取图像
img = cv2.imread('defect2.png')

# 转为浮点数类型的数据
data = np.float32(img.reshape((-1,3)))

# 聚类
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 8
ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 将每个像素点的标签转换为图像矩阵
label = label.reshape((img.shape[0], img.shape[1]))
label = np.array(label, dtype=np.uint8)
# 显示图像
cv2.imshow('image', label*30)
cv2.waitKey(0)
#cv2.destroyAllWindows()


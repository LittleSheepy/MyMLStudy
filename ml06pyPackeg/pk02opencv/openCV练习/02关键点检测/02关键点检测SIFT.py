import cv2
import numpy as np
# 读取图片
image_path = r'D:\04DataSets\04\box.jpg'
img = cv2.imread(image_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建SIFT特征提取器对象
sift = cv2.xfeatures2d.SIFT_create()
# 检测关键点
keypoints = sift.detect(gray_img, None)
# 绘制关键点
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 显示结果
cv2.imshow('Image with Keypoints', img_with_keypoints)
cv2.waitKey()
cv2.destroyAllWindows()
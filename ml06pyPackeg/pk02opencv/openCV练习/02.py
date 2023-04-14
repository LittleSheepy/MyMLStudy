
import cv2
import numpy as np
dir_root = r"D:\02dataset\01work\01TuoPanLJ\tuopan\0tmp_test/"
img1 = cv2.imread('1.bmp')
img2 = cv2.imread('2.bmp')
#2. 实现图片的匹配，即找到两个图片的共同点并将它们重叠在一起。可以使用SIFT算法或SURF算法来实现这一步骤。


sift = cv2.xfeatures2d.SIFT_create()
#找到关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
#建立FLANN匹配器
flann = cv2.FlannBasedMatcher()
#匹配关键点
matches = flann.knnMatch(des1, des2, k=2)
# 筛选算法
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
#根据这些点，计算透视变换
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#3. 进行图像融合，将两个图片拼接在一起。


# 图像融合
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
dst1 = cv2.perspectiveTransform(pts1, M)
dst = np.concatenate((dst1, pts2), axis=0)
[x_min, y_min] = np.int32(dst.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(dst.max(axis=0).ravel() + 0.5)
t = [-x_min, -y_min]
H = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
result = cv2.warpPerspective(img1, H.dot(M), (x_max - x_min, y_max - y_min))
result[t[1]:h2 + t[1], t[0]:w2 + t[0]] = img2
#4. 显示和保存最终结果。


cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('panorama.jpg', result)
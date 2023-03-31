
import cv2
import numpy as np
# 读入要拼接的图像
dir_root = r"D:\02dataset\01work\01TuoPanLJ\tuopan\0tmp_test/"
img01_path = dir_root + "1.bmp"
img02_path = dir_root + "2.bmp"
img1 = cv2.imread(img01_path)
img2 = cv2.imread(img02_path)
# 初始化ORB对象，并使用OR算法进行特征提取
orb = cv2.ORB_create()
# 提取图像1和图像2的关键点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# 使用FLANN算法进行特征匹配
index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# 选择距离小于0.7倍最近邻距离的匹配项
good_matches = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)
# 从匹配项中获取关键点的坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# 使用RANSAC算法计算单应性矩阵，并过滤掉错误匹配
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matches_mask = mask.ravel().tolist()
# 计算图像拼接后的大小，以确保图像不会被裁剪
h, w = img1.shape[:2]
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)
xmin, ymin = np.int32(dst.min(axis=0).ravel() - 0.5)
xmax, ymax = np.int32(dst.max(axis=0).ravel() + 0.5)
# 使用单应性矩阵进行图像拼接
tx, ty = -xmin, -ymin
M[0, 2] += tx
M[1, 2] += ty
result = cv2.warpPerspective(img1, M, (xmax-xmin, ymax-ymin))
result[ty:ty+h, tx:tx+w] = img2
# 显示结果
cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyAllWindows()
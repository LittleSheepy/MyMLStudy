import cv2
import numpy as np
# 读取两张图片
dir_root = r"D:\02dataset\01work\01TuoPanLJ\tuopan\0tmp_test3/"
img01_path = dir_root + "1.jpg"
img02_path = dir_root + "2.jpg"
img1 = cv2.imread(img01_path)
img2 = cv2.imread(img02_path)
# 提取ORB特征和描述符
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# 使用暴力匹配器进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
# 将匹配点按距离排序
matches = sorted(matches, key=lambda x:x.distance)
# 保留前50个匹配点
matches = matches[:50]
# 提取匹配点的位置
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
# 计算透视变换矩阵
M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
# 对img2进行透视变换
warp = cv2.warpPerspective(img2, M, (img1.shape[1]+img2.shape[1], img1.shape[0]))
# 将img1和变换后的img2拼接在一起
warp[0:img1.shape[0], 0:img1.shape[1]] = img1
# 显示拼接后的图像
cv2.imwrite('result.jpg', warp)
cv2.imshow('result', warp)
cv2.waitKey(0)
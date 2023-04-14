# 导入库
import cv2
import numpy as np
import sys
from PIL import Image


def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


dir_root = r"D:\02dataset\01work\01TuoPanLJ\tuopan\0tmp_test3/"
img1_path = dir_root + "2.bmp"
img2_path = dir_root + "1.bmp"
ima = cv2.imread(img1_path)
imb = cv2.imread(img2_path)
A = ima.copy()
B = imb.copy()
imageA = cv2.resize(A, (0, 0), fx=0.8, fy=0.7)
imageB = cv2.resize(B, (0, 0), fx=0.9, fy=0.7)


def detectAndDescribe(image):
    sift = cv2.SIFT_create()
    (kps, features) = sift.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    return kps, features


kpsA, featuresA = detectAndDescribe(imageA)
kpsB, featuresB = detectAndDescribe(imageB)
bf = cv2.BFMatcher()
matches = bf.knnMatch(featuresA, featuresB, 2)
good = []
for m in matches:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
        good.append((m[0].trainIdx, m[0].queryIdx))

if len(good) > 4:
    ptsA = np.float32([kpsA[i] for (_, i) in good])
    ptsB = np.float32([kpsB[i] for (i, _) in good])
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

M = (matches, H, status)
if M is None:
    print("无匹配结果")
    sys.exit()
(matches, H, status) = M
result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
show('res', result)
print(result.shape)
cv2.imwrite('./123.jpg', result)
print("ok")
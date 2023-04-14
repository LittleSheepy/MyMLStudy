import cv2
import os

# 定义函数进行sift检测并画图
def sift_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.06)
    gray = cv2.blur(gray, (5, 5))
    gray = cv2.blur(gray, (5, 5))
    kp, des = sift.detectAndCompute(gray, None)
    # kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray, kp, image)
    return img

# 遍历文件夹
root_dir = r"D:\04DataSets\ningjingLG\02ZangWu\LateralPollution/"
input_folder = root_dir + 'img/'
output_folder = root_dir + 'sift/'
for filename in os.listdir(input_folder):
    # 读取图片
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    # 进行sift检测并画图
    sift_img = sift_detect(img)
    # 保存图片
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, sift_img)

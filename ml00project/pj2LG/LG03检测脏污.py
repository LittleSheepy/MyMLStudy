import cv2
import os
import numpy as np

# 定义输入输出文件夹路径
root_dir = r"D:\04DataSets\ningjingLG\02ZangWu\LateralPollution/"
input_folder = root_dir + 'img/'
output_folder = root_dir + 'template_blur_threshold2_left/'
output_folder_img = root_dir + 'template_blur_threshold2_left_img/'

# 读取模版图片灰度图
template = cv2.imread(root_dir + 'black_0075290_CM2_2_t_left.bmp', cv2.IMREAD_GRAYSCALE)
h, w = template.shape[0], template.shape[1]
# 遍历文件夹
for filename in os.listdir(input_folder):
    if filename[-5] != "2":
        continue
    # 读取灰度图
    img_bgr = cv2.imread(os.path.join(input_folder, filename))
    img_gray = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)

    # 模版匹配
    img_gray_ = img_gray[:, 0:img_gray.shape[1]//2]
    res = cv2.matchTemplate(img_gray_, template, cv2.TM_CCOEFF_NORMED)
    max_ = res.max()
    # 画出匹配结果
    threshold = max_
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # 保存在另一个文件夹
    cv2.imwrite(os.path.join(output_folder_img, filename), img_bgr)
    xmin = loc[1][0] + 100
    xmax = xmin + 600
    ymin = 700 + 100
    ymax = 1670 - 100
    box_center = img_gray[ymin:ymax, xmin:xmax]
    img_blur = box_center.copy()
    box_center = cv2.blur(box_center, (7, 7))
    img_blur = cv2.blur(box_center, (600, 600))
    img_del = abs(box_center.astype(np.float64) - img_blur.astype(np.float64))
    img_del = img_del.astype(np.uint8)
    img_del = cv2.blur(img_del, (5, 5))
    ret, thresh = cv2.threshold(img_del, 25, 255, cv2.THRESH_BINARY)    # +cv2.THRESH_OTSU
    #box_center = cv2.cvtColor(box_center, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, filename), cv2.hconcat([box_center, thresh]))








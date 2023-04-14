import os
import cv2

# 定义输入输出文件夹路径
root_dir = r"D:\04DataSets\ningjingLG\02ZangWu\LateralPollution/"
input_folder = root_dir + 'img/'
output_folder = root_dir + 'threshold180/'

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 构建文件路径
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # 读取图像并转换为灰度图
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    thresh = 180
    # if filename[0] == "g":
    #     thresh = 210
    # 对图像进行二值化处理
    ret, thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    # 保存处理后的图像
    cv2.imwrite(output_path, thresh)





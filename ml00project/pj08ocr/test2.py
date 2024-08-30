import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'D:\02dataset\01work\11OCR\03lableimg_one\img_little/LOT-MARKING_1_20240820140000000_N1AWSH303_FEDDH20AN1_OT8IoT00008iMV_SIDE_1_1_OK_OK_999131.jpg')

# 获取图像尺寸
height, width = image.shape[:2]

# 创建波浪扭曲效果
def wave_distortion(image, amplitude, frequency):
    rows, cols = image.shape[:2]
    distorted_image = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            # 计算波浪效果
            offset = int(amplitude * np.sin(2 * np.pi * frequency * (i / rows)))
            if j + offset < cols and j + offset >= 0:
                distorted_image[i, j] = image[i, j + offset]
            else:
                distorted_image[i, j] = image[i, j]  # 边界处理

    return distorted_image

# 应用波浪扭曲
amplitude = 20  # 波幅
frequency = 5   # 频率
wavy_image = wave_distortion(image, amplitude, frequency)

# 显示结果
cv2.imshow('Wavy Distortion', wavy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
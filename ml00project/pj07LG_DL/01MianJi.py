import cv2
import numpy as np


def filter_color_range(image, lower_bound, upper_bound):
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建掩膜，只保留指定范围的颜色
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # 将掩膜应用到原始图像上
    result = cv2.bitwise_and(image, image, mask=mask)

    # 将不在范围内的颜色置黑
    black_background = np.zeros_like(image)
    black_background[mask != 0] = result[mask != 0]

    return black_background


# 示例用法
if __name__ == "__main__":
    image = cv2.imread('img.bmp')
    lower_bound = np.array([110, 130, 150])  # 颜色范围下界
    upper_bound = np.array([160, 200, 200])  # 颜色范围上界

    filtered_image = filter_color_range(image, lower_bound, upper_bound)
    cv2.imwrite('output_image.jpg', filtered_image)

import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'D:\02dataset\01work\11OCR\03lableimg_one\img_little/LOT-MARKING_1_20240820140000000_N1AWSH303_FEDDH20AN1_OT8IoT00008iMV_SIDE_1_1_OK_OK_999131.jpg')

# 获取图像尺寸
height, width = image.shape[:2]

# 定义随机扭曲的控制点
# src_points = np.float32([[0, height], [width, height], [0, 0], [width, 0]])
# dst_points = np.float32([
#     [np.random.randint(-50, 50), np.random.randint(height - 50, height)],  # 左下角
#     [np.random.randint(width - 50, width), np.random.randint(height - 50, height)],  # 右下角
#     [np.random.randint(-50, 50), np.random.randint(0, 50)],  # 左上角
#     [np.random.randint(width - 50, width), np.random.randint(0, 50)]  # 右上角
# ])
# 定义扭曲的控制点
src_points = np.float32([[0, height], [width, height], [0, 0], [width, 0]])
dst_points = np.float32([
    [0, height],  # 左下角不变
    [width, height - 50],  # 右下角向上扭曲
    [0, 0],  # 左上角不变
    [width, 0]  # 右上角不变
])

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 应用透视变换
distorted_image = cv2.warpPerspective(image, matrix, (width, height))

# 显示结果
cv2.imshow('Randomly Distorted Image', distorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
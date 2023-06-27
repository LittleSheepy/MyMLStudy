import numpy as np
import cv2

# 创建一个全白的图像，尺寸为 300x300，数据类型为 uint8
white_image = np.full((440, 440), 255, dtype=np.uint8)
cv2.rectangle(white_image, (10, 10), (430, 430), 0, -1)
cv2.rectangle(white_image, (20, 20), (420, 420), 200, -1)
x0 = 20
y0 = 20
x1 = 420
y1 = 420
# 定义线的颜色（BGR格式）和粗细
color = 0  # 绿色
thickness = 2
for i in range(17):
    # 横线
    # 定义线的起点和终点
    start_point = (x0, y0+i*25)
    end_point = (420, y0+i*25)
    cv2.line(white_image, start_point, end_point, color, thickness)
    # 定义线的起点和终点
    start_point = (x0+i*25, y0)
    end_point = (x0+i*25, 420)
    cv2.line(white_image, start_point, end_point, color, thickness)

# 保存图像为灰度图
cv2.imwrite('gray_image.jpg', white_image)



# 侧面
len = 521
side_image = np.full((85+40, len+40), 255, dtype=np.uint8)
cv2.rectangle(side_image, (10, 10), (len+30, 85+30), 0, -1)
cv2.rectangle(side_image, (20, 20), (len+20, 85+20), 200, -1)

cv2.line(side_image, (20, 40), (len+20, 40), 0, 2)

distence = [0, 40, 18, 64, 56, 56, 53, 56, 56, 64, 18, 40]
x0 = 20
y0 = 40
for i in range(12):
    # 定义线的起点和终点
    x0 = x0 + distence[i]
    start_point = (x0, y0)
    end_point = (x0, 85+20)
    cv2.line(side_image, start_point, end_point, color, thickness)


cv2.imwrite('gray_image1.jpg', side_image)




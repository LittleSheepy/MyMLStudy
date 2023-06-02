import cv2
from PIL import Image
import numpy as np

#img_path = r"D:\02dataset\01work\05nanjingLG\01imgall\0LG_DATA\052120_052208\imgAll\NM/原始图black_0056702_NM1_1.jpg"
img_path_zh = r"测试.jpg"
img_path_en = r"test.jpg"
# 读取图像
pil_image = Image.open(img_path_zh)
# 将图像转换为numpy数组
numpy_image = np.array(pil_image)
# 将numpy数组转换为opencv格式
opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
cv2.imshow("opencv_image", opencv_image)
cv2.waitKey(0)
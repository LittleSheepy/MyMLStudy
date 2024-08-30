# -*- coding: utf-8 -*-
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 加载图像
image = cv2.imread(r"D:\02dataset\01work\11OCR\04char_class_all\0_result\R\LOT-MARKING_1_20240820140001000_N1AWSH303_FEDDH20AN1_OT8IoT00008iNZ_SIDE_1_1_OK_OK_99913523.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# 获取图片的高度和宽度
height, width = image.shape[:2]

# 计算最短边
shortest_edge = max(height, width)

# 计算需要填充的像素数
padding_height = (shortest_edge - height) // 2
padding_width = (shortest_edge - width) // 2

# 填充图片
image = cv2.copyMakeBorder(image, padding_height, padding_height, padding_width, padding_width,
                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
# 定义增强操作
transform = A.Compose([
    A.RandomResizedCrop(height=224, width=224,
        scale=(1.0, 1.0),
        ratio=(0.95, 1.0 / 0.95))
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),
    # A.RandomBrightnessContrast(p=0.2),
])

# 进行增强
transformed = transform(image=np.array(image))

# 获取增强后的图像
transformed_image = transformed['image']

# 可视化增强后的图像
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image)
axs[0].set_title('src image')
axs[0].axis('off')

axs[1].imshow(transformed_image)
axs[1].set_title('transformed image')
axs[1].axis('off')

plt.show()
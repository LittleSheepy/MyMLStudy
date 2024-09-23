import cv2, os
import numpy as np


def pad_to_square(image):
    h, w = image.shape[:2]
    if h == w:
        return image

    size = max(h, w)
    pad_color = [0, 0, 0]  # White padding

    if h > w:
        pad_right = size - w
        padded = cv2.copyMakeBorder(image, 0, 0, 0, pad_right, cv2.BORDER_CONSTANT, value=pad_color)
    else:
        pad_bottom = size - h
        padded = cv2.copyMakeBorder(image, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=pad_color)

    return padded

if __name__ == '__main__':
    dir_root = r"D:\02dataset\10coco128\coco128_1\images/"
    dir_name = "train2017"
    dir_src = dir_root + dir_name
    dir_save = dir_root + dir_name + "_pad"
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)
    for img_name in os.listdir(dir_src):
        img_path = os.path.join(dir_src, img_name)
        image = cv2.imread(img_path)
        square_image = pad_to_square(image)
        cv2.imwrite(os.path.join(dir_save, img_name), square_image)


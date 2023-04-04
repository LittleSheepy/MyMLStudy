import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

import psutil
import gc
import cvSeg




def save_pre_mask(img_gray, template_gray):
    h, w = img_gray.shape[0], img_gray.shape[1]
    loc, score = cvSeg.matchTemplate(img_gray, template_gray, 0.2)
    ymin = min(loc[0])
    ymax = max(loc[0])
    xmin = min(loc[1])
    xmax = max(loc[1])
    img_gray_mask = np.zeros(img_gray.shape)
    img_gray_mask[ymin:ymax,xmin:xmax] = 255
    plt.imshow(img_gray_mask)
    plt.show()
    currentAxis = plt.gca()
    plt.imshow(img_gray)
    rect = plt.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), color="blue", fill=False, linewidth=1)
    currentAxis.add_patch(rect)
    plt.show()

    pass


if __name__ == '__main__':
    imgName = "CB_00001.png"
    global_imagename = imgName
    dir_root = r"D:\04DataSets\02/"
    dir_images = dir_root + r"\images/"
    dir_mask = dir_root + r"\mask/"
    dir_images_max_template = dir_root + r"\images_max_template_sqdiff/"
    dir_images_flg2 = dir_root + r"\images_flg23/"
    img_path = dir_images + imgName
    mask_path = dir_mask + imgName[:-4] + "_t" + ".bmp"
    template_path = "./template.png"
    template2_path = "./template2.png"
    template2CenterBlack_path = "./template2CenterBlack.png"
    template3_path = "./template3.png"
    defect2_path = "./defect2.png"
    defect2Left_path = "./defect2Left.png"
    img_bgr = cv2.imread(img_path)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask_global = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template2_gray = cv2.imread(template2_path, cv2.IMREAD_GRAYSCALE)
    defect2_gray = cv2.imread(defect2_path, cv2.IMREAD_GRAYSCALE)
    defect2Left_gray = cv2.imread(defect2Left_path, cv2.IMREAD_GRAYSCALE)
    save_pre_mask(img_gray, template_gray)

    loc = matchTemplate(defect2_gray, template_gray)
    det = template_nms(img_gray, template_gray, 0.5)
    det = sorted(det, key=lambda x: (x[1], x[0]))
    save_det(img_bgr, det)
    pass



import cv2, os
import numpy as np

def dm2_2little(img_gray):
    _, binary_img = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectList = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        rectList.append(rect)
    return rectList

def dir_dm2_2little():
    for filename in os.listdir(dir_dm2):
        print(filename)
        filepath = os.path.join(dir_dm2, filename)
        img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.imread(filepath)
        rectList = dm2_2little(img_gray)
        for i in range(len(rectList)):
            x, y, w, h = rectList[i]
            # 截取外接矩形的图像
            roi = img_bgr[y:y + h, x:x + w]

            filepath_new = os.path.join(dir_dm2_little, filename[:-4] + "_" + str(i) + ".jpg")
            # 保存图像
            cv2.imwrite(filepath_new, roi)


if __name__ == '__main__':
    dir_root = "./"
    dir_dm2 = dir_root + "DM2/"
    dir_dm2_little = dir_root + "DM2_little/"

    if not os.path.exists(dir_dm2_little):
        os.makedirs(dir_dm2_little)

    dir_dm2_2little()

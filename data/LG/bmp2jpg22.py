import cv2, os
import numpy as np


def bmp2jpg(dir_SRC, dir_SRC_JPG):
    for name in os.listdir(dir_SRC):
        dir_name_SRC = os.path.join(dir_SRC, name)
        if os.path.isdir(dir_name_SRC):
            print(name)
            dir_name_SRC_JPG = os.path.join(dir_SRC_JPG, name)
            if not os.path.exists(dir_name_SRC_JPG):
                os.mkdir(dir_name_SRC_JPG)
            bmp2jpg(dir_name_SRC, dir_name_SRC_JPG)
        else:
            name_jpg = name[:-4] + ".bmp"
            dir_filename_SRC_JPG = os.path.join(dir_SRC_JPG, name_jpg)
            if not os.path.exists(dir_filename_SRC_JPG):
                cv2.imwrite(dir_filename_SRC_JPG, cv2.imread(dir_name_SRC))

if __name__ == '__main__':
    dir_root = r"./"
    dir_SRC = dir_root + "SRC/"
    dir_SRC_JPG = dir_root + "SRC_JPG/"
    if not os.path.exists(dir_SRC_JPG):
        os.mkdir(dir_SRC_JPG)
    bmp2jpg(dir_SRC, dir_SRC_JPG)
    print("结束")
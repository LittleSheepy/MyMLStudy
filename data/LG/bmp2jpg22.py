import cv2, os
import numpy as np

from PIL import Image

def convert_bmp_to_jpg(bmp_file_path, jpg_file_path):
    try:
        with Image.open(bmp_file_path) as img:
            img = img.convert('RGB')
            img.save(jpg_file_path, 'JPEG')
        print(f"Successfully converted {bmp_file_path} to {jpg_file_path}")
    except Exception as e:
        print(f"Failed to convert image: {e}")

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
            name_jpg = name[:-4] + ".jpg"
            dir_filename_SRC_JPG = os.path.join(dir_SRC_JPG, name_jpg)
            if not os.path.exists(dir_filename_SRC_JPG):
                # cv2.imwrite(dir_filename_SRC_JPG, cv2.imread(dir_name_SRC))
                convert_bmp_to_jpg(dir_name_SRC, dir_filename_SRC_JPG)
if __name__ == '__main__':
    # dir_root = r"./"
    dir_root = r"H:\0426-更新模型后点检数据/"
    dir_SRC = dir_root + "原图_改名/"
    dir_SRC_JPG = dir_root + "原图_改名_JPG/"
    if not os.path.exists(dir_SRC_JPG):
        os.mkdir(dir_SRC_JPG)
    bmp2jpg(dir_SRC, dir_SRC_JPG)
    print("结束")
import cv2, os
import numpy as np
import shutil

def is_top_third_gray_white(image_path):
    image = cv2.imread(image_path)
    height, _, _ = image.shape
    top_third = image[0:height//5, :]

    gray_white_pixels = np.sum((top_third >= [170, 170, 170]) & (top_third <= [255, 255, 255]))
    total_pixels = np.prod(top_third.shape)

    if gray_white_pixels / total_pixels >= 0.9:
        return True
    else:
        return False

def is_center_green(image_path):
    image = cv2.imread(image_path)
    height, _, _ = image.shape
    top_third = image[height//5:height-height//5, :]

    # gray_white_pixels = np.sum((top_third >= [50, 125, 111]) & (top_third <= [80, 175, 160]))
    green_pixels = 0

    height, width, _ = top_third.shape
    for y in range(height):
        for x in range(width):
            r, g, b = top_third[y, x]
            if g > r and g > b:
                green_pixels += 1
    total_pixels = height * width

    if green_pixels / total_pixels >= 0.2:
        return True
    else:
        return False




def main():
    for file_name in os.listdir(img_src):
        img_path = os.path.join(img_src, file_name)
        if is_top_third_gray_white(img_path) and is_center_green(img_path):
            shutil.copy(img_path, img_fen_white + file_name)
        else:
            shutil.copy(img_path, img_fen_other + file_name)





if __name__ == '__main__':
    dir_root = r"D:/"
    dir_mame = "test2"
    img_src = os.path.join(dir_root, dir_mame)
    img_fen = os.path.join(dir_root, dir_mame+"_fen")
    img_fen_white = img_fen + "/white/"
    img_fen_other = img_fen + "/other/"
    os.makedirs(img_fen, exist_ok=True)
    os.makedirs(img_fen_white, exist_ok=True)
    os.makedirs(img_fen_other, exist_ok=True)
    main()


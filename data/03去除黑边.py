import os
import cv2
import numpy as np
from scipy.stats import mode
import time
import concurrent.futures

'''
    multi-process to crop pictures.
'''


def crop(file_path_list):
    origin_path, save_path = file_path_list
    img = cv2.imread(origin_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    closed_1 = cv2.erode(gray, None, iterations=4)
    closed_1 = cv2.dilate(closed_1, None, iterations=4)
    blurred = cv2.blur(closed_1, (9, 9))
    # get the most frequent pixel
    num = mode(blurred.flat)[0][0] + 1
    # the threshold depends on the mode of your images' pixels
    num = num if num <= 30 else 1

    _, thresh = cv2.threshold(blurred, num, 255, cv2.THRESH_BINARY)

    # you can control the size of kernel according your need.
    kernel = np.ones((13, 13), np.uint8)
    closed_2 = cv2.erode(thresh, kernel, iterations=4)
    closed_2 = cv2.dilate(closed_2, kernel, iterations=4)

    _, cnts, _ = cv2.findContours(closed_2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the image
    # cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
    # cv2.imshow("Image", img)
    # cv2.imwrite("pic.jpg", img)
    # cv2.waitKey(0)

    xs = [i[0] for i in box]
    ys = [i[1] for i in box]
    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)
    height = y2 - y1
    width = x2 - x1
    crop_img = img[y1:y1 + height, x1:x1 + width]
    cv2.imwrite(save_path, crop_img)
    # cv2.imshow("Image", crop_img)
    # cv2.waitKey(0)
    print(f'the {origin_path} finish crop, most frequent pixel is {num}')


def multi_process_crop(input_dir):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(crop, input_dir)

def run_crop(input_dir):
    for file_path_list in input_dir:
        crop(file_path_list)



if __name__ == "__main__":
    data_dir = r'E:\0ProjectData\0LG_CB_DATA\1AIDI_TrainData\0LG_label_name\BM\16BM_JQJPS/'
    save_dir = r'E:\0ProjectData\0LG_CB_DATA\1AIDI_TrainData\0LG_label_name\BM\16BM_JQJPS1/'
    path_list = [(os.path.join(data_dir, o), os.path.join(save_dir, o)) for o in os.listdir(data_dir)]
    start = time.time()
    run_crop(path_list)
    print(f'Total cost {time.time() - start} seconds')
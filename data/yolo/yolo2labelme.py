# coding:utf-8

import os
import cv2
import json
import matplotlib.pyplot as plt
import base64


def parse_tta_label(txt_path, img_dir, save_dir):
    file_name = txt_path.split('/')[-1].split('.')[0]
    img_path = os.path.join(img_dir, file_name + ".jpg")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(img_path, 'rb') as f:
        image = f.read()
    image_base64 = str(base64.b64encode(image), encoding='utf-8')

    with open(txt_path, 'r') as f:
        label_info_list = f.readlines()

    version = "4.5.13"
    data_dict = dict()
    data_dict.__setitem__("version", version)
    data_dict.__setitem__("imagePath", file_name)
    data_dict.__setitem__("imageData", image_base64)
    data_dict.__setitem__("imageHeight", h)
    data_dict.__setitem__("imageWidth", w)
    data_dict.__setitem__("flags", {})
    data_dict["shapes"] = list()
    for label_info in label_info_list:
        label_info = label_info.strip()
        label_info = label_info.split(' ')
        class_name = label_info[0]
        point_cnt = int((len(label_info)-1)/2)
        points = []
        for i in range(point_cnt):
            px = label_info[1+2*i]
            py = label_info[1+2*i+1]
            x = int(float(px) * w)
            y = int(float(py) * h)
            points.append([x,y])

        shape_type = "polygon"
        shape = {}

        shape.__setitem__("label", "WR")
        shape.__setitem__("points", points)
        shape.__setitem__("shape_type", shape_type)
        shape.__setitem__("flags", {})
        shape.__setitem__("group_id", None)
        data_dict["shapes"].append(shape)

    save_json_path = os.path.join(save_dir, "%s.json" % file_name)
    json.dump(data_dict, open(save_json_path, 'w'), indent=4)
    #### for view debug
    # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # plt.figure(0)
    # plt.imshow(img)
    # plt.show()


def generate_labelme_prelabel(txt_dir, img_dir, save_dir):
    txt_name_list = os.listdir(txt_dir)
    for txt_name in txt_name_list:
        if txt_name.startswith('.'):
            continue
        print("processing -> %s" % txt_name)
        txt_path = os.path.join(txt_dir, txt_name)
        parse_tta_label(txt_path, img_dir, save_dir)


if __name__ == '__main__':
    root_dir = r"D:\05xx\0518\/"
    txt_dir = root_dir + "txt/"
    save_dir = root_dir + "json/"

    img_dir = root_dir + r"\img\/"

    generate_labelme_prelabel(txt_dir, img_dir, save_dir)


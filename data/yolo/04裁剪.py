import os
import cv2
import json


def cut_img():
    for img_name in os.listdir(big_img_dir):
        file_name = img_name.split('/')[-1].split('.')[0]
        img_path = os.path.join(big_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        cut_x_left = 1060
        cut_y_left = 940
        cut_w_left = 1210
        cut_h_left = 2260
        img_cut = img[940:940+cut_h_left, 1060:1060+cut_w_left,:]
        save_img_path = os.path.join(small_img_dir, img_name)
        cv2.imwrite(save_img_path, img_cut)

        json_path = os.path.join(big_json_dir, file_name + ".json")
        with open(json_path, "r") as r:
            json_info = json.load(r)
        shapes=json_info["shapes"]
        for shape in shapes:
            shape["points"][0][0] = shape["points"][0][0] - 1060
            shape["points"][0][1] = shape["points"][0][1] - 940
            shape["points"][1][0] = shape["points"][1][0] - 1060
            shape["points"][1][1] = shape["points"][1][1] - 940

        save_json_path = os.path.join(small_json_dir, "%s.json" % file_name)
        json.dump(json_info, open(save_json_path, 'w'), indent=4)


        pass


if __name__ == '__main__':
    root_dir = r"C:\Users\KADO\Desktop\ZZS\C\/"
    big_root_dir = root_dir + "INSPECTION/"
    big_img_dir = big_root_dir + "img/"
    big_txt_dir = big_root_dir + "txt/"
    big_json_dir = big_root_dir + "json/"
    small_root_dir = root_dir + "INSPECTION_small/"
    small_img_dir = small_root_dir + "img/"
    small_txt_dir = small_root_dir + "txt/"
    small_json_dir = small_root_dir + "json/"
    cut_img()
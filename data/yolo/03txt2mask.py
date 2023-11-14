import os
import cv2
import numpy as np


# 用于单标签
# 返回yolov5生成txt中的实际点位，以及与原始图片大小相同的纯色画布
def getdata(img_path, txt_path):
    img = cv2.imread(img_path)  # 读取图片信息
    img_x = img.shape[0]
    img_y = img.shape[1]
    with open(txt_path, "r") as f:  # 打开文件
        data_content = f.read()  # 读取文件
    data_list = data_content.split('\n')
    data = []
    for data_item in data_list:
        d = data_item.split(' ', -1)
        # d[-1] = d[-1][0:-1]
        data_poly = []
        for i in range(1, int(len(d) / 2) + 1):
            if d[2 * i - 1] == "":
                break
            data_poly.append([img_y * float(d[2 * i - 1]), img_x * float(d[2 * i])])
        data_poly = np.array(data_poly, dtype=np.int32)
        data.append(data_poly)

    img = np.zeros((img_x, img_y, 1))  # 黑色背景
    return data, img


def txt2mask():
    files = os.listdir(img_dir)
    for file in files:
        name = file[0:-4]
        img_path = img_dir + '/' + name + '.jpg'
        txt_path = txt_dir + '/' + name + '.txt'
        data_list, img = getdata(img_path, txt_path)
        color = 1
        for data in data_list:
            if data.size <= 0:
                break
            cv2.fillPoly(img,  # 原图画板
                         [data],  # 多边形的点
                         color=color)
        save_path = mask_dir + '/' + name + '.png'
        cv2.imwrite(save_path, img[:,:,0])

if __name__ == '__main__':
    root_dir = r"E:\0ProjectData\0LG_CB_DATA\2labelData\01NM_LZPS\1all\/"
    img_dir = root_dir + '/img/'
    txt_dir = root_dir + '/txt/'
    mask_dir = root_dir + "/mask/"
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    txt2mask()

import os
import cv2

def readTxt(txt_path, w, h):
    rectList = []
    if os.path.isfile(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                data = line.split(" ")
                cls = data[0]
                cx = float(data[1])*w
                cy = float(data[2])*h
                dw = float(data[3])*w
                dh = float(data[4])*h
                lx = cx - (dw/2)
                ly = cy - (dh/2)
                rtRect = [lx, ly, dw, dh]
                rectList.append(rtRect)
    return rectList

def drawBox(img, rectList):
    for rect in rectList:
        cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
                  (0, 255, 0), 2)
    return img

def main():
    for filename in os.listdir(img_dir):
        img_name = filename[:-4]
        img_label = cv2.imread(img_dir+filename)
        img_predict = cv2.imread(img_predict_path + filename)
        img_pre = cv2.imread(img_dir+filename)
        h, w = img_pre.shape[1], img_pre.shape[0]

        txt_label_path = txt_label + img_name + ".txt"
        rectList_label = readTxt(txt_label_path, w, h)
        img_label = drawBox(img_label, rectList_label)

        txt_pre_path = txt_pre + img_name + ".txt"
        rectList_pre = readTxt(txt_pre_path, w, h)
        img_pre = drawBox(img_pre, rectList_pre)

        result_img = cv2.hconcat([img_label, img_pre, img_predict])
        cv2.imwrite(img_save + filename, result_img)



if __name__ == '__main__':
    root_dir = r"D:\0\0LG_DATA\01coco128little0606\/"
    img_dir = root_dir + r"img_have/"
    img_predict_path = root_dir + r"img_pre/"
    img_save = root_dir + r"img_duibi/"
    txt_label = root_dir + r"txt_train/"
    txt_pre = root_dir + r"txt_labels/"
    main()
import os
import cv2
import copy
import pickle
import numpy as np
import xml.etree.ElementTree as ET


def main():
    out_dir = "D:/01sheepy/01work/01baojie_ocr/pp/xml_txt/"
    xml_dir = "D:/01sheepy/01work/01baojie_ocr/pp/xml/"
    for name in os.listdir(xml_dir):
        print(name)
        filename = os.path.join(xml_dir, name)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        f = open(os.path.join(out_dir, name[:-4]+".txt"), "a")
        for obj in objs:
            transcription = obj.find('name').text
            bbox = obj.find('bndbox')
            x1 = bbox.find('xmin').text
            y1 = bbox.find('ymin').text
            x2 = bbox.find('xmax').text
            y2 = bbox.find('ymax').text
            f.write(x1+","+y1+","+x2+","+y1+","+x2+","+y2+","+x1+","+y2+","+transcription+"\n")
        f.close()

def rec_trans():
    img_dir = "D:/01sheepy/01work/01baojie_ocr/pp/img/"
    xml_dir = "D:/01sheepy/01work/01baojie_ocr/pp/xml/"
    out_dir = "D:/01sheepy/01work/01baojie_ocr/pp/word_new/"
    count = 1

    f = open(os.path.join(out_dir, "rec_gt_train.txt"), "a")
    f_test = open(os.path.join(out_dir, "rec_gt_test.txt"), "a")
    for name in os.listdir(xml_dir):
        print(name)
        filename = os.path.join(xml_dir, name)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        imgname = os.path.join(img_dir, name[:-4]+".jpg")
        img = cv2.imread(imgname)
        for obj in objs:
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = int(float(bbox.find('xmin').text))
            y1 = int(float(bbox.find('ymin').text))
            x2 = int(float(bbox.find('xmax').text))
            y2 = int(float(bbox.find('ymax').text))
            imgtmp = img[y1:y2, x1:x2,:]
            img_path_train = "train/pendian_" + str(count) + ".jpg"
            img_path_test = "train/pendian_" + str(count) + ".jpg"
            cv2.imwrite(out_dir + "train/pendian_"+str(count)+".jpg", imgtmp)
            label = label.replace("\"", "")
            f.write(img_path_train + '\t' + label + '\n')
            f_test.write(img_path_test + '\t' + label + '\n')
            count += 1
    f.close()
    f_test.close()


if __name__ == '__main__':
    main()
    #rec_trans()













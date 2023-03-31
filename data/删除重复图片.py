import os, shutil,sys,getopt,random
import cv2
import copy
import pickle
import numpy as np
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd
import cv2


def quchong():
    img_dir = "/home/rex/PythonProgram/yolov5/sevencode_data/train_imgs/"
    biaozhu = "/home/rex/PythonProgram/yolov5/sevencode_data/images/"
    num=0
    for imgfile in os.listdir(biaozhu):
        moveFlg = False
        for xmlfile in os.listdir(img_dir):
            # print(xmlfile)
            if xmlfile[:-4] == imgfile[:-4]:
                # shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
                os.remove(img_dir+xmlfile)
                num+=1
                moveFlg = True
    print(num)
        # if not moveFlg:
        # print(imgfile)


def fenshuju():
    img_dir = "/home/rex/PythonProgram/yolov5/sevencode_data/val_images_1_20/"
    biaozhu = "/home/rex/PythonProgram/yolov5/sevencode_data/val_images_xmls/"
    # nobiaozhu = "test_data/image_nobiaozhu/"
    # xml_dir = "test_data/image_biaozhu_train_xml"
    i = 0
    for imgfile in os.listdir(img_dir):
        img_path=img_dir+imgfile
        xml_path=biaozhu+imgfile[:-4]+".xml"

        if i % 5==0:
            shutil.copyfile(img_path, "/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/5/"+imgfile)
            shutil.copyfile(xml_path, "/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/5_xml/"+imgfile[:-4]+".xml")

        if i % 5==1:
            shutil.copyfile(img_path, "/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/1/" + imgfile)
            shutil.copyfile(xml_path,"/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/1_xml/" + imgfile[:-4] + ".xml")

        if i % 5==2:
            shutil.copyfile(img_path, "/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/2/" + imgfile)
            shutil.copyfile(xml_path,"/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/2_xml/" + imgfile[:-4] + ".xml")

        if i % 5==3:
            shutil.copyfile(img_path, "/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/3/" + imgfile)
            shutil.copyfile(xml_path,"/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/3_xml/" + imgfile[:-4] + ".xml")

        if i % 5==4:
            shutil.copyfile(img_path, "/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/4/" + imgfile)
            shutil.copyfile(xml_path,"/home/rex/PythonProgram/yolov5/sevencode_data/jiancha/4_xml/" + imgfile[:-4] + ".xml")
        i+=1
        # for xmlfile in os.listdir(xml_dir):
        #     print(xmlfile)
        #     if xmlfile[:-4] == imgfile[:-4]:
        #         shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
        #         moveFlg = True
        # if not moveFlg:
        #     shutil.copyfile(img_dir + imgfile, nobiaozhu + imgfile)
        pass
    pass


def txt_to_img():
    """将图片和xml分出来"""
    txt_path="/home/rex/PythonProgram/yolov5/sevencode_data/val.txt"
    imgs_path="/home/rex/PythonProgram/yolov5/sevencode_data/images/"
    xml_path="/home/rex/PythonProgram/yolov5/sevencode_data/Annotations/"
    num=0
    with open(txt_path,"r") as f:
        contents=f.readlines()
        for line in contents:
            num+=1
            line=line.strip()
            listline=line.split("\t")[0]
            one_step=listline.split("/")[-1]
            shutil.copyfile(imgs_path + one_step, "/home/rex/PythonProgram/yolov5/sevencode_data/val_imgs/" + one_step)
            shutil.copyfile(xml_path + one_step[:-4]+".xml", "/home/rex/PythonProgram/yolov5/sevencode_data/val_xmls/" + one_step[:-4]+".xml")

            print(one_step)
    print(",,,,,,,,,,,,,,,,,,,,,",num)
    pass


classes = ["0","1","2", "3","4", "5","6", "7","8", "9","A", "b","C", "d","E", "F","J","h","P","t", "U", "u", "_"]   # 改成自己的类别


def get_label(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    num = 0
    num2 = 0
    label_bbox={}

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b_value=int(xmlbox.find('xmin').text)
        label_bbox[b_value]=cls
        # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
        #      float(xmlbox.find('ymax').text))

   # print(label_bbox)
    bb_list=sorted(label_bbox.keys())
    #print(sorted(label_bbox.keys()))
    str=''
    for bb in bb_list:
        str+=label_bbox[bb]
    print(str)
    return str
    pass


def img_label_to_txt():
    """将图片个对应的label写入txt"""
    txt_path="/home/rex/PythonProgram/yolov5/sevencode_data/txt/val_imgs_labels.txt"
    images_path="/home/rex/PythonProgram/yolov5/sevencode_data/images/"
    xmls_path="/home/rex/PythonProgram/yolov5/sevencode_data/image_xml/"
    out_file = open(txt_path, 'w')
    for file in os.listdir(images_path):
        xml_path=xmls_path+file[:-4]+".xml"
        print(file)
        label=get_label(xml_path)
        cont=file+';'+label+'\n'
        out_file.write(cont)
    out_file.close()
    pass


def write_img_path_to_txt():
    """在txt中追加文本内容"""
    txt_path = "/home/rex/PythonProgram/yolov5/sevencode_data/val.txt"
    out_file = open(txt_path, 'a')
    imgs_path="/home/rex/PythonProgram/yolov5/sevencode_data/kone"
    for img_path in os.listdir(imgs_path):
        print(img_path)
        img_path="/home/rex/PythonProgram/yolov5/sevencode_data/images/"+img_path
        out_file.write(img_path+"\n")
        pass
    out_file.close()


if __name__ == '__main__':
    quchong()
    #fenshuju()
    #txt_to_img()
    #img_label_to_txt()
    #write_img_path_to_txt()
    pass
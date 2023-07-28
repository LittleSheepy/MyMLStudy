import os, shutil,sys,getopt,random
import cv2
import copy
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import json
"""函数功能描述"""
#1、将原始数据分为训练集和测试集：RandPickTrainTestData()
#2、将xml文件转成txt文件
#3、将

def move1():
    root_dir = r"E:\0ProjectData\0LG_CB_DATA\1AIDI_TrainData\0LG_label_name\nmjqjps\train\/"
    img_dir = root_dir + "/img/"
    biaozhu = root_dir + "/img_have/"
    nobiaozhu = root_dir + "/image_nobiaozhu/"
    xml_dir = root_dir + "/txt_have/"
    num1=0
    num2=0
    for imgfile in os.listdir(img_dir):
        num1+=1
        moveFlg = False
        for xmlfile in os.listdir(xml_dir):
            num2+=1
            #print(xmlfile)
            if xmlfile[:-4] == imgfile[:-4]:
                #shutil.move(img_dir + imgfile, biaozhu + imgfile)
                shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
                moveFlg = True
        #print(num2)
        # if moveFlg:
        #     print(imgfile)
        #     shutil.move(img_dir + imgfile, biaozhu + imgfile)
    # print(num1)


def check_xml_label():
    """检查标注文件是否有标注信息 """
    img_dir = "test_data/image_all/"
    xml_dir = "test_data/xml_all"
    for name in os.listdir(xml_dir):
        # print(name)
        filename = os.path.join(xml_dir, name)
        # domtree=parse(filename)
        # objects=domtree.documentElement
        # objs=objects.getElementsByTagName('object')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not objs:
            print(objs)
            print(name)
            # f = open(os.path.join(out_dir, name[:-4] + ".txt"), "a")
            # for obj in objs:
            #     # transcription = obj.getElementsByTagName('name')[0].childNodes[0].data
            #     transcription = obj.find('name').text
            #     bbox = obj.find('bndbox')
            #     print(bbox)


def move_noxmlpic():
    """移除没有xml标注文件的图片"""
    dir="test_data/image_biaozhu_train/"
    save_path="test_data/image_nobiaozhu/"
    image_list=[]
    xml_list=[]

    for file in os.listdir(dir):
        path = dir + file
        print(path)
        if file.endswith(".jpg"):
            image_list.append(file)
        else:
            xml_list.append(file)
    for iamge_path in image_list:       #one_b_error001_2835.jpg'
        moveFlg = False
        for xml_path in xml_list:
            if xml_path[:-4] == iamge_path[:-4]:
                moveFlg = True

        if not moveFlg:
            shutil.copyfile("test_data/image_biaozhu_train/" + iamge_path, save_path + iamge_path)
            os.remove("test_data/image_biaozhu_train/" + iamge_path)


def move_xml():
    xml_dir="test_data/image_biaozhu_xml/"
    xml_test_dir="test_data/image_biaozhu_test_xml/"
    xml_save_dir="test_data/image_biaozhu_train_xml/"

    test_xml_list=[]
    for test_xml in os.listdir(xml_test_dir):
        test_xml_list.append(test_xml[:-4])

    for xml in os.listdir(xml_dir):
        if xml[:-4] not in test_xml_list:
            shutil.copyfile(xml_dir + xml, xml_save_dir + xml)
            pass


def RandPickTrainTestData():
    """随机将数据分成训练集和测试集"""
    """parames:traindata，testdata,trainxml,testxml,rate"""
    dir_root = r"D:/01sheepy/01work/02tongllidianti/0dataset/imgall/"
    train_dir=dir_root + "train/img/"
    test_dir = dir_root + "test/img/"
    trainxml_dir = dir_root + "train/xml/"
    testxml_dir = dir_root + "test/xml/"
    rate=0.2
    rate = float(rate)
    pathDir = os.listdir(train_dir)
    filenumber = len(pathDir)
    print("filenumber = ", filenumber)
    picknumber = int(filenumber * rate)
    print("picknumber = ", picknumber)
    sample = random.sample(pathDir, picknumber)
    for name in sample:
        shutil.move(os.path.join(trainxml_dir, name[:-4]+".xml"), os.path.join(testxml_dir, name[:-4]+".xml"))
        shutil.move(os.path.join(train_dir, name), os.path.join(test_dir, name))
        print(name)
        pass
    return


def gen_det_label(input_dir, out_label):
    """针对OCR det"""
    """由xml生成label_txt文件"""
    """parames:图片路径，xml路径,label_txt路径"""
    root_path="icdar_c4_train_imgs/"
    input_dir=""
    with open(out_label, 'w') as out_file:
        for label_file in os.listdir(input_dir):
            img_path = root_path + label_file[:-4] + ".jpg"
            label = []
            with open(os.path.join(input_dir, label_file), 'r') as f:
                for line in f.readlines():
                    tmp = line.strip("\n\r").replace("\xef\xbb\xbf",
                                                     "").split(',')
                    points = tmp[:8]
                    s = []
                    for i in range(0, len(points), 2):
                        b = points[i:i + 2]
                        b = [int(float(t)) for t in b]
                        s.append(b)
                    result = {"transcription": tmp[8], "points": s}
                    label.append(result)

            out_file.write(img_path + '\t' + json.dumps(
                label, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    #move_xml()
    move1()
    #move_noxmlpic()
    # check_xml_label()
    #RandPickTrainTestData()
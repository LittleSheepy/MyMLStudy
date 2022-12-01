
import os, shutil,sys,getopt,random
import cv2
import copy
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import json

def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id), mode='rb')
    try:
        tree = ET.parse(in_file)
    except Exception as e:
        return
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(float(bndbox.find('xmin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymin = int(float(bndbox.find('ymin').text))
        ymax = int(float(bndbox.find('ymax').text))
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)
    if root.find('object') is None:
        return None
    bndbox = root.find('object').find('bndbox')
    return bndboxlist

def imgAddXML():
    dir_root = r"D:\01sheepy\01work\06ningbo\01img\04热检缺陷\/"
    dir_save = r"D:\01sheepy\01work\06ningbo\01img\04热检缺陷\zl_画框\/"
    dir_img = dir_root+"/zl_标注的/"
    dir_xml = dir_root+"/zlxml/"
    for imgfile in os.listdir(dir_img):
        print(imgfile)
        xmlfile = imgfile[:-3]+"xml"
        #img = cv2.imread(dir_img+imgfile)
        img = cv2.imdecode(np.fromfile(dir_img+imgfile, dtype=np.uint8), -1)
        try:
            bndboxlist = read_xml_annotation(dir_xml, xmlfile)
        except:
            continue

        # 绘制一个红色矩形
        ptLeftTop = (bndboxlist[0][0], bndboxlist[0][1])
        ptRightBottom = (bndboxlist[0][2], bndboxlist[0][3])
        point_color = (0, 0, 255) # BGR
        thickness = 5
        lineType = cv2.LINE_8
        cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        imgsavepath = dir_save+imgfile
        #imgsavepath = imgsavepath.encode('gbk')
        #imgsavepath = imgsavepath.decode()
        cv2.imencode('.jpg', img)[1].tofile(imgsavepath)
        #cv2.imwrite(imgsavepath,img)

if __name__ == '__main__':
    imgAddXML()
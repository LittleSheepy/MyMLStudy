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

def creatTxt():
    root_dir = r"D:\05xxNMJQJPS\noadd\left_0805-10\/"
    right_str = "_src0.3"
    img_dir = root_dir + "/img" + right_str + "/"
    txt_dir = root_dir + "/txt" + right_str + "/"
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    for imgfile in os.listdir(img_dir):
        f = open(txt_dir + imgfile[:-4] + ".txt", "a")
        f.close()

creatTxt()
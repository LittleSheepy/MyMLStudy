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
    root_dir = r"G:\06LG\0LG_DATA\SZ_TEST_OK0527\/"
    img_dir = root_dir + "/img_save11_have0604/"
    txt_dir = root_dir + "/txt_save11_have0604/"
    for imgfile in os.listdir(img_dir):
        f = open(txt_dir + imgfile[:-4] + ".txt", "a")
        f.close()

creatTxt()
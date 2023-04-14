
"""
data is list of label
lable = {
    "index"   : index,  # 图片编号/名字
    "bboxes"  : bboxes,
    "classes" : classes
}
"""
import cv2
import time
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tf_slim as slim

from lib.config.Config import config as config
# from lib.datasets.pascalVocDS import pascalVocDS as pascalVocDS
from lib.tools.utils import *
from lib.tools.bbox import *



def ds_func_SSD(data):
    inputs = []
    classes_data = []
    locs_data = []
    scores_data = []
    anchors = generate_anchors_by_size_ratios()
    grids = generate_grids(config.strides)
    for label in data:
        img_index = label["index"]
        bboxes = np.array(label["bboxes"])

        classes = label["classes"]
        img_path = config.data_path + "JPEGImages/" + img_index + ".jpg"
        img = cv2.imread(img_path)
        img_shape = img.shape
        bboxes_norm = bboxes / [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        processed_img = preprocess_image(img, (300, 300), "resize")
        inputs.append(processed_img[0])
        for i in range(6):
            if len(classes_data) <= i:
                classes_data.append([])
                locs_data.append([])
                scores_data.append([])
            feature_shape = [config.feature_shapes[i], config.feature_shapes[i]]
            feat_classes, feat_locs, feat_scores = bboxes_encode(classes, bboxes_norm, anchors[i],grids[i],
                                                                 feature_shape)
            classes_data[i].append([feat_classes])
            locs_data[i].append([feat_locs])
            scores_data[i].append([feat_scores])
    return inputs, locs_data, scores_data, classes_data

def ds_func_read_img(img_names):
    inputs = []
    for img_name in img_names:
        img = cv2.imread(img_name)
        processed_img = preprocess_image(img, (416, 416), "resize")
        inputs.append(processed_img[0])
    return inputs





















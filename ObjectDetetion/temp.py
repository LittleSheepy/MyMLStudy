import tensorflow as tf
import numpy as np
import pprint
import cv2
from lib.datasets.pascalVocDS import pascalVocDS

import matplotlib.pyplot as plt
from lib.config.Config import config as config
import math
from lib.datasets.pascalVocDS import pascalVocDS as pascalVocDS
from lib.tools.utils import *
from lib.tools.bbox import *

data_train = pascalVocDS()

labels = data_train.next_batch(config.batch_size)
inputs = []
classes_data = []
locs_data = []
scores_data = []
anchors = generate_anchors_by_size_ratios()
for label in labels:
    img_index = label["index"]
    bboxes = np.array(label["bboxes"])

    classes = label["classes"]
    img_path = config.data_path + "JPEGImages/" + img_index + ".jpg"
    img = cv2.imread(img_path)
    img_shape = img.shape
    bboxes_norm = bboxes/ [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    processed_img = preprocess_image(img, (300, 300), "resize")
    inputs.append(processed_img)
    for i in range(6):
        if len(classes_data) <= i:
            classes_data.append([])
            locs_data.append([])
            scores_data.append([])
        feature_shape = [config.feature_shapes[i],config.feature_shapes[i]]
        feat_labels, feat_localizations, feat_scores = bboxes_encode(classes, bboxes_norm, anchors[i], feature_shape)
        classes_data[i].append(feat_labels)
        locs_data[i].append(feat_localizations)
        scores_data[i].append(feat_scores)



a = 1
[[0.14164601 0.6130187  0.29442495 0.88704556], [0.1417957  0.6055622  0.29607636 0.89571655], [0.1394405  0.60971373 0.28853858 0.88583785], [0.14645152 0.60952413 0.29356462 0.9043921 ], [0.15431452 0.62136275 0.30414492 0.906365  ]]

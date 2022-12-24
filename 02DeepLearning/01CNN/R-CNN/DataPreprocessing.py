# 数据预处理
import os
import tflearn
import codecs
import cv2
import numpy as np

from AlexNet import create_alexnet
import config
import util
from util import resize_image, IOU, clip_pic
from SelectiveSearch import SelectiveSearch

def image_proposal(img_path):
    img = cv2.imread(img_path)

    img_lbl, regions = SelectiveSearch(img, scale=2000, sigma=0.9, min_size=10).selective_search()
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding small regions
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 10000 or (r['rect'][2] * r['rect'][3]) >= (img.shape[0]-1) * (img.shape[1]-1):
            continue
        # resize to 227 * 227 for input
        proposal_img, proposal_vertice = util.clip_pic(img, r['rect'])
        # Delete Empty array
        if len(proposal_img) == 0:
            continue
        # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        # Check if any 0-dimension exist
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = util.resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices

def load_train_proposals(datafile, num_clss, save_path, threshold=0.5, is_svm=False, save=False):
    fr = open(datafile, 'r')
    train_list = fr.readlines()
    # random.shuffle(train_list)
    labels = []
    images = []
    for num, line in enumerate(train_list):
        tmp = line.strip().split(' ')
        # tmp0 = image address
        # tmp1 = label
        # tmp2 = rectangle vertices
        img = cv2.imread(config.DATA_PATH + tmp[0])
        img_lbl, regions = SelectiveSearch(img, scale=2000, sigma=0.9, min_size=10).selective_search()
        candidates = set()
        max_iou = 0
        for r in regions:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding small regions
            if r['size'] < 220:
                continue
            if (r['rect'][2] * r['rect'][3]) < 10000 or (r['rect'][2] * r['rect'][3]) >= (img.shape[0]-1) * (img.shape[1]-1):
                continue
            # resize to 227 * 227 for input
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            # Delete Empty array
            if len(proposal_img) == 0:
                continue
            # Ignore things contain 0 or not C contiguous array
            x, y, w, h = r['rect']
            if w == 0 or h == 0:
                continue
            # Check if any 0-dimension exist
            [a, b, c] = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:
                continue
            resized_proposal_img = resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
            candidates.add(r['rect'])
            img_float = np.asarray(resized_proposal_img, dtype="float32")
            images.append(img_float)
            # IOU
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = IOU(ref_rect_int, proposal_vertice)
            #print("iou_val", iou_val)
            if max_iou < iou_val:
                max_iou = iou_val
            # labels, let 0 represent default class, which is background
            index = int(tmp[1])
            if is_svm:
                if iou_val < threshold:
                    labels.append(0)
                else:
                    labels.append(index)
            else:
                label = np.zeros(num_clss + 1)
                if iou_val < threshold:
                    label[0] = 1
                else:
                    label[index] = 1
                labels.append(label)
        util.view_bar("processing image of %s" % datafile.split('\\')[-1].strip(), num + 1, len(train_list))
    if save:
        np.save(save_path, [images, labels])
    fr.close()
    return images, labels

# load data
def load_from_npy(data_set):
    images, labels = [], []
    data_list = os.listdir(data_set)
    # random.shuffle(data_list)
    for ind, d in enumerate(data_list):
        i, l = np.load(os.path.join(data_set, d), allow_pickle=True)
        images.extend(i)
        labels.extend(l)
        util.view_bar("load data of %s" % d, ind + 1, len(data_list))
    print(' ')
    return images, labels

#### 17类预训练数据 预处理##########
def preprocessingPreTrainData(save=False):
    save_path = config.DATA_PATH + 'PreTrainData.npy'
    fr = codecs.open(config.TRAIN_LIST, 'r', 'utf-8')
    train_list = fr.readlines()
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        fpath = tmp[0]
        img = cv2.imread(config.DATA_PATH + fpath)
        img = util.resize_image(img, config.IMAGE_SIZE, config.IMAGE_SIZE)
        np_img = np.asarray(img, dtype="float32")
        images.append(np_img)

        index = int(tmp[1])
        label = np.zeros(config.TRAIN_CLASS)
        label[index] = 1
        labels.append(label)
    if save:
        np.save(save_path, [images, labels])
    fr.close()
    return images, labels

def loadPreTrainData():
    save_path = config.DATA_PATH + 'PreTrainData.npy'
    if os.path.isfile(save_path):
        X, Y = np.load(save_path, allow_pickle=True)
        return list(X), list(Y)
    return preprocessingPreTrainData()

##### 2类数据微调训练 预处理#######
def preprocessingFineTuneData(save=False):
    processed_data_path = config.DATA_PATH + "fineTuneData.npy"
    X, Y = load_train_proposals(config.FINE_TUNE_LIST, 2, save=save, save_path=processed_data_path)
    return X, Y

def loadFineTuneData():
    save_path = config.DATA_PATH + "fineTuneData.npy"
    if os.path.isfile(save_path):
        X, Y = np.load(save_path, allow_pickle=True)
        return list(X), list(Y)
    return preprocessingFineTuneData()


#### SVM数据 预处理##########
def preprocessingSVMdata(save=False):
    train_file_folder = config.TRAIN_SVM
    for train_file in os.listdir(train_file_folder):
        file_split = train_file.split('.')
        if file_split[-1] == 'txt':
            save_path = os.path.join(train_file_folder, file_split[0]) + "_svmData.npy"
            train_file = os.path.join(train_file_folder, train_file)
            load_train_proposals(train_file, 2, save_path, threshold=0.3, is_svm=True, save=save)

def loadSVMdata(data_path):
    if not os.path.isfile(data_path):
        preprocessingSVMdata(save=True)
    X, Y = np.load(data_path, allow_pickle=True)
    return list(X), list(Y)

def preprocessingData():
    preprocessingPreTrainData(True)
    preprocessingFineTuneData(True)
    preprocessingSVMdata(True)

if __name__ == '__main__':
    preprocessingData()







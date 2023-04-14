import os
import cv2
import copy
import pickle
import numpy as np
import xml.etree.ElementTree as ET

from .DataSetBase import DataSetBase
from lib.config.Config import config as config

class pascalVocDS(DataSetBase):
    def __init__(self):
        super(pascalVocDS, self).__init__()
        self.dsDir = config.data_path
        self.nameFile = os.path.join(self.dsDir, "ImageSets\Main", config.ds["imgNameListFile"])
        self.cls_names = config.voc_classes
        self.labels = self.loadLables()
        self.num_label = len(self.labels)

    def loadLables(self):
        cache_file = config.cache_file
        if os.path.isfile(cache_file):
            print('Loading cache_file from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                labels = pickle.load(f)
            return labels

        print('Processing labels from: ' + self.dsDir)

        with open(self.nameFile) as f:
            self.imgIndexes = [x.strip() for x in f.readlines()]

        labels = []
        for index in self.imgIndexes:
            bboxes, classes = self.load_bboxes(index)
            lable = {
                "index"   : index,
                "bboxes"  : bboxes,
                "classes" : classes
            }
            labels.append(lable)
        with open(cache_file, "wb") as f:
            pickle.dump(labels, f)
        return labels

    def load_bboxes(self, index):
        """Load bboxes from XML file."""
        imname = os.path.join(self.dsDir, 'JPEGImages', index + '.jpg')
        filename = os.path.join(self.dsDir, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        size_obj = tree.find("size")
        w = float(size_obj.find("width").text)
        h = float(size_obj.find("height").text)
        objs = tree.findall('object')
        bboxes = []
        classes = []
        for obj in objs:
            cls = self.cls_to_ind[obj.find('name').text.lower().strip()]
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
            classes.append(cls)
        return bboxes, classes

    #def load







if __name__ == '__main__':
    ds = pascalVocDS()
    ds.loadLables()











import os
import cv2
import copy
import pickle
import numpy as np

from .DataSetBase import DataSetBase
from lib.config.Config import config as config

class imgsDS(DataSetBase):
    def __init__(self):
        super(imgsDS, self).__init__()
        self.dsDir = config.data_path
        self.img_names = self.get_img_names()
        self.labels = self.loadLables()

    def loadLables(self):
        img_names = []
        for file in os.listdir(self.dsDir):
            file_path = os.path.join(self.dsDir, file)
            if not os.path.isdir(file_path):
                img_names.append(file_path)
        return img_names




if __name__ == '__main__':
    imgds = imgsDS()
    imgds.dsDir = "D:\ML_datas\coco\coco2014/train2014/"
    imgds.get_img_names()
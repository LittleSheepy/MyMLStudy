import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from algseg.dataset.seg_dataset_base import SegDatasetBase


class Chase_db1(SegDatasetBase):
    def __init__(self, root, split='training', transform=None, target_suffix="_1stHO"):
        self.data_root = root
        self.transform = transform
        self.split = split
        self.target_suffix = target_suffix

        # self.images_dir = os.path.join(self.data_root, 'images', self.split)
        # self.targets_dir = os.path.join(self.data_root, 'annotations', self.split)
        self.images_dir = self.data_root
        self.targets_dir = self.data_root
        self.images = []
        self.targets = []
        if split not in ['training', 'test', 'validation']:
            raise ValueError('Invalid split for mode! Please use split="training", split="test"'
                             ' or split="validation"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        for file_name in os.listdir(self.images_dir):
            if "_1stHO" in file_name or "_2ndHO" in file_name:
                continue
            self.images.append(os.path.join(self.images_dir, file_name))
            target_name = '{}{}.{}'.format(file_name.split('.')[0],
                                         self.get_target_suffix(), "png")
            self.targets.append(os.path.join(self.targets_dir, target_name))

    def get_target_suffix(self):
        return self.target_suffix



if __name__ == '__main__':
    from algseg.utils import ext_transforms as et
    data_root = r"D:\02dataset\CHASEDB1/"
    transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtRandomCrop(size=(512, 512)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    data = Chase_db1(data_root, transform=transform)
    data0 = data[0]
    pass
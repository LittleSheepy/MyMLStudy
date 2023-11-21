import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class SegDatasetBase(data.Dataset):
    def __init__(self, root, split='training', transform=None):
        self.data_root = root
        self.transform = transform
        self.split = split

        self.images_dir = os.path.join(self.data_root, "images", split)
        self.targets_dir = os.path.join(self.data_root, "annotations", split)
        self.images = []
        self.targets = []

        for file_name in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(os.path.join(self.targets_dir, file_name))

    @classmethod
    def encode_target(cls, target):
        target[target > 0] = 1
        return target

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)

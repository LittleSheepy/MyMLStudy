import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import Cityscapes
from algseg.utils import ext_transforms as et

from algseg.dataset import Cityscapes
"""
Folder structure
data
--gtFine
----train
------aachen
--------aachen_000000_000019_gtFine_labelIds
----test
----val
--leftImg8bit
----train
------aachen
--------aachen_000000_000019_leftImg8bit
----test
----val
"""

#
# class CCityScapes_torchvision():
#     def __init__(self, root):
#         self.root = root
#         self.transform = transforms.Compose([
#             transforms.Resize((512, 512)),  # 调整图像大小
#             transforms.RandomCrop(size=(512, 512)),
#             transforms.ToTensor(),  # 转换为Tensor格式
#             transforms.Normalize((0.485, 0.456, 0.406),
#                                  (0.229, 0.224, 0.225))  # 图像标准化
#         ])
#         self.tar_transform = transforms.Compose([
#             transforms.Resize((512, 512)),  # 调整图像大小
#             transforms.RandomCrop(size=(512, 512)),
#             transforms.PILToTensor(),  # 转换为Tensor格式
#         ])
#         self.train_dataset = None
#         self.train_dataloader = None
#         self.batch_size = 2
#
#     def init_dataset(self, dataset_types):
#         if "train" in dataset_types:
#             self.train_dataset = Cityscapes(
#                 root=self.root, split='train', mode='fine', target_type='semantic',
#                 transform=self.transform, target_transform=self.tar_transform)
#
#             self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
#




class CCityScapes():
    def __init__(self, root):
        self.data_root = root

        self.train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(512, 512)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None
        self.batch_size = 2

    def init_dataset(self, dataset_types):
        if "train" in dataset_types:
            self.train_dataset = Cityscapes(root=self.data_root,
                               split='train', transform=self.train_transform)

            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        if "val" in dataset_types:
            self.val_dataset = Cityscapes(root=self.data_root,
                               split='val', transform=self.val_transform)

            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size*2, shuffle=False)

        if "test" in dataset_types:
            self.test_dataset = Cityscapes(root=self.data_root,
                               split='test', transform=self.val_transform)

            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size*2, shuffle=False)



















if __name__ == '__main__':
    data_root = r"F:\sheepy\00MyMLStudy\ml10Repositorys\05open-mmlab\mmsegmentation-1.2.1_my\data\cityscapes/"
    dataset = CCityScapes(data_root)
    dataset.init_dataset(["train"])
    train_dataloader = dataset.train_dataloader
    for batch in train_dataloader:
        pass





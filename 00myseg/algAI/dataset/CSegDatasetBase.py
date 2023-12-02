from torch.utils.data import DataLoader
from algseg.utils import ext_transforms as et
from algseg.dataset.seg_dataset_base import SegDatasetBase

"""
Folder structure
data
--images
----train
--------*.png
----test
----val
--annotations
----train
--------*.png
----test
----val
"""
class CSegDatasetBase():
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
            self.train_dataset = SegDatasetBase(root=self.data_root,
                               split='training', transform=self.train_transform)

            self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        if "val" in dataset_types:
            self.val_dataset = SegDatasetBase(root=self.data_root,
                               split='validation', transform=self.val_transform)

            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size*2, shuffle=False)

        if "test" in dataset_types:
            self.test_dataset = SegDatasetBase(root=self.data_root,
                               split='test', transform=self.val_transform)

            self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size*2, shuffle=False)


if __name__ == '__main__':
    data_root = r"D:\02dataset\CHASEDB1/"
    dataset = CSegDatasetBase(data_root)
    dataset.init_dataset(["train"])
    train_dataloader = dataset.train_dataloader
    for batch in train_dataloader:
        pass


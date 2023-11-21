"""
参考https://github.com/milesial/Pytorch-UNet
"""
from algseg.config.CCfgBase import CCfgBase


class CCfgUNet(CCfgBase):
    def __init__(self):
        super().__init__()
        self.data_channels = 3
        self.bilinear = False
        self.data_root = r"D:\02dataset\CHASEDB1/"



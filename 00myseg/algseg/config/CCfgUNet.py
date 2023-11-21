from algseg.config.CCfgBase import CCfgBase


class CCfgUNet(CCfgBase):
    def __init__(self):
        super().__init__()
        self.n_channels = 3
        self.n_classes = 1000
        self.bilinear = False
        self.data_root = r"F:\sheepy\00MyMLStudy\ml10Repositorys\05open-mmlab\mmsegmentation-1.2.1_my\data\cityscapes/"



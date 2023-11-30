from algseg.config.CCfgBase import CCfgBase


class CCfgExample(CCfgBase):
    def __init__(self):
        super().__init__()
        self.n_channels = 3
        self.n_classes = 1000
        self.bilinear = False
        self.data_root = r"./data/"



from cmodels.CModelBase import CModelBase, CCfgBase
from models.unet import


class CUNet(CModelBase):
    def __init__(self, config: CCfgBase):
        super(CUNet).__init__(config)
        # self.model =
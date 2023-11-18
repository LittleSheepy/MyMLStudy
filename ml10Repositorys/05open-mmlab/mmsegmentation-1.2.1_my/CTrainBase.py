"""
batch_size, 数字
train_epoch, 数字
model_type, ["seg", "det", "cls"]
model_size, ["large", "middle", "small"]
model_name, string
outpyt_dir, string

"""
class CCfgBase():
    def __init__(self, **para):
        for key, value in para.items():
            setattr(self, key, value)
    # def check_para(self):
    #     para_require = ["batch_size", "train_epoch"]
    #     for key, value in para.items():
    #         setattr(self, key, value)


class CTrainBase:
    def __init__(self):
        self.show_data = {}
        
    def train(self, cfg: CCfgBase):
        pass


import train
class CCfgMMSEG():
    def __init__(self, **para):
        super(CCfgMMSEG).__init__(**para)
        self.config = r"configs\unet/unet_s5-d16_deeplabv3_4xb4-40k_chase-db1-128x128.py"
        self.work_dir = r"./output/unet_s5-d16_deeplabv3_4xb4-40k_chase-db1-128x128_train1024_01"
        

class CTrainMMSEG(CTrainBase):
    def train(self, cfg: CCfgMMSEG):
        train.main(cfg.config, cfg.work_dir)


if __name__ == '__main__':
    cfgmmseg = CCfgMMSEG()
    mmseg = CTrainMMSEG()
    mmseg.train(cfgmmseg)

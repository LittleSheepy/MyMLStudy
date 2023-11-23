from algseg.cmodels.CUNet import CUNet, CCfgUNet

if __name__ == '__main__':
    cfg = CCfgUNet()
    model = CUNet(cfg)
    model.train()
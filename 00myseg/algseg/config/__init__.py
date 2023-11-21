from algseg.config.CCfgUNet import CCfgUNet


def get_cfg(cfg_nage):
    if "unet" == cfg_nage:
        return CCfgUNet()
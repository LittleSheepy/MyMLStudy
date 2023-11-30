from algseg.config.CCfgUNet import CCfgUNet
from algseg.config.CCfgExample import CCfgExample


def get_cfg(cfg_nage):
    if "unet" == cfg_nage:
        return CCfgUNet()
    elif "example" == cfg_nage:
        return CCfgExample()


if __name__ == '__main__':
    print(get_cfg("example"))


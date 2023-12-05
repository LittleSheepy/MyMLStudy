

class CfgBase:
    def __init__(self):
        self.name = "CfgBase"

class CfgA(CfgBase):
    def __init__(self):
        super().__init__()
        self.name = "CfgA"
        self.paraA = "A"


class MyCls:
    def __init__(self, cfg: CfgBase):
        print(cfg.paraA)

if __name__ == '__main__':
    cfg = CfgA()
    mycls = MyCls(cfg)